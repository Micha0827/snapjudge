"""Browser agent: snapjudge picks the next browser action, Playwright executes it.

Observe -> decide -> act, like Jev Browser / Jev Ultrafast: the page's interactive elements are
numbered (e0, e1, ...), and ONE snapjudge request answers two typed questions about the same
page state -- "is the task done?" (noul) and "which action next?" (choice). Every choice option is
a concrete action on one element ("e3: type into the field 'Produktsuche' and press Enter"),
plus scroll and back. Picking operation and element separately (Jev's operation x target split)
confused small models: they answered DONE or SCROLL on the first page. The page state is
prefilled once (state_first layout), both questions reuse it. Text is generated only for typing,
and by the same local model, so no second LLM is needed.

    python agent/browser_agent.py --model <mlx model> [--adapter dir] \\
        --url https://example.org --task "..." [--max-steps 15] [--headful]

    python agent/browser_agent.py --url file://.../shop.html --dry-run   # show what the model sees

Every run writes a trace (one JSON line per step, with timings and probabilities) and the final
screenshot to results/browser/<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NEEDS_TARGET = {"CLICK", "TYPE", "TYPE_ENTER", "SELECT"}
TEXT_KINDS = {"input:text", "input:search", "input:email", "input:tel", "input:url", "input:number",
              "input:password", "textarea"}
DONE_THRESHOLD = 0.5
# Consent banners are never the model's call: reject non-essential cookies up front, and never
# offer "accept all" as an action.
REJECT_LABELS = ["Alle ablehnen", "Reject all", "Ablehnen", "Nur notwendige", "Only necessary"]
ACCEPT_ALL = re.compile(r"(alle akzeptieren|accept all|alle annehmen|allow all|alle zulassen)", re.I)

# Numbers every visible interactive element (data-sj="eN") and describes it in one line.
EXTRACT_JS = r"""
(maxEls) => {
  document.querySelectorAll('[data-sj]').forEach(e => e.removeAttribute('data-sj'));
  const sel = 'a[href], button, input:not([type=hidden]), select, textarea, [role=button], ' +
              '[role=link], [role=tab], [role=checkbox], [role=option], [role=menuitem], [role=combobox], ' +
              '[role=radio], [role=switch], [role=gridcell], [onclick], summary';
  const vh = window.innerHeight, out = [];
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  const cands = [...document.querySelectorAll(sel)].filter(el => {
    const r = el.getBoundingClientRect(), st = getComputedStyle(el);
    if (!(r.width > 0 && r.height > 0 && st.visibility !== 'hidden' && st.display !== 'none' && !el.disabled)) return false;
    // skip elements covered by something else (open dropdowns, dialogs): check the topmost
    // element at the center of the ones that are in the viewport
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    if (cy > 0 && cy < vh && cx > 0 && cx < window.innerWidth) {
      const top = document.elementFromPoint(cx, cy);
      if (top && top !== el && !el.contains(top) && !top.contains(el)) return false;
    }
    return true;
  });
  // open suggestions first, then elements in the viewport, then the ones below it
  const rank = el => {
    const r = el.getBoundingClientRect();
    if (el.getAttribute('role') === 'option') return 0;
    return r.top < vh && r.bottom > 0 ? 1 : 2;
  };
  cands.sort((a, b) => rank(a) - rank(b) || a.getBoundingClientRect().top - b.getBoundingClientRect().top);
  for (const el of cands.slice(0, maxEls)) {
    const id = 'e' + out.length;
    el.setAttribute('data-sj', id);
    const tag = el.tagName.toLowerCase();
    const type = el.getAttribute('type');
    const kind = tag === 'input' ? 'input:' + (type || 'text') : (el.getAttribute('role') || tag);
    const focused = el === document.activeElement;
    const parts = [];
    const label = el.labels && el.labels[0] ? clean(el.labels[0].innerText) : '';
    const text = clean(el.innerText || '').slice(0, 80);
    if (label) parts.push('label "' + label + '"');
    if (text && tag !== 'select') parts.push('"' + text + '"');
    for (const a of ['aria-label', 'placeholder', 'name', 'title', 'alt']) {
      const v = clean(el.getAttribute(a));
      if (v && !parts.some(p => p.includes(v))) parts.push(a + '="' + v.slice(0, 60) + '"');
    }
    // a label one level down (e.g. calendar days: "1" outside, "Donnerstag, 1. Oktober 2026" inside)
    if (!el.getAttribute('aria-label')) {
      const inner = el.querySelector('[aria-label]');
      const v = inner ? clean(inner.getAttribute('aria-label')) : '';
      if (v && !parts.some(p => p.includes(v))) parts.push('aria-label="' + v.slice(0, 60) + '"');
    }
    if (tag === 'input' || tag === 'textarea') {
      if (type === 'checkbox' || type === 'radio') parts.push(el.checked ? 'checked' : 'unchecked');
      else if (el.value) parts.push('value="' + el.value.slice(0, 60) + '"');
    }
    let options = null;
    if (tag === 'select') {
      options = [...el.options].map(o => clean(o.text));
      parts.push('selected "' + clean(el.options[el.selectedIndex]?.text) + '"');
      parts.push('options: ' + options.slice(0, 12).join(' | ') + (options.length > 12 ? ' | ...' : ''));
    }
    if (tag === 'a') {
      const href = el.getAttribute('href') || '';
      if (!text && href) parts.push('href=' + href.slice(0, 60));
    }
    const r = el.getBoundingClientRect();
    if (r.top >= vh) parts.push('(below the fold)');
    // Enter submits a form only when it has a single text field (a search box)
    const textSel = 'input[type=text], input[type=search], input:not([type]), textarea';
    const submits = !!(el.form && el.form.querySelectorAll(textSel).length === 1 && tag !== 'textarea');
    if (focused) parts.push('(focused)');
    out.push({id, kind, desc: parts.join(', '), options, submits});
  }
  return {
    url: location.href,
    title: document.title,
    text: clean(document.body ? document.body.innerText : '').slice(0, 2500),
    elements: out,
  };
}
"""


PAGE_TEXT_CHARS = 1500


def page_state(task: str, obs: dict, history: list[str]) -> str:
    """The STATE snapjudge judges. The element list is not repeated here: the action options
    already describe every element, and the state is the part prefilled on every step."""
    lines = [
        f"TASK: {task}",
        "",
        f"URL: {obs['url']}",
        f"TITLE: {obs['title']}",
        "",
        "ACTIONS SO FAR:",
        *(history[-8:] or ["(none)"]),
        "",
        "VISIBLE PAGE TEXT (truncated):",
        obs["text"][:PAGE_TEXT_CHARS] or "(empty)",
    ]
    return "\n".join(lines)


def element_action(e: dict) -> tuple[str, str]:
    """The one natural action on an element, as (operation, description for the model)."""
    if e["kind"] in TEXT_KINDS:
        if e.get("submits"):
            return "TYPE_ENTER", f"type text into the field {e['desc']} and press Enter"
        return "TYPE", f"type text into the field {e['desc']}"
    if e["kind"] == "option":
        return "CLICK", f"pick the suggestion {e['desc']}"
    if e["kind"] == "select":
        return "SELECT", f"choose an entry in the dropdown {e['desc']}"
    if e["kind"] in ("input:checkbox", "input:radio", "checkbox"):
        return "CLICK", f"toggle the {e['kind'].split(':')[-1]} {e['desc']}"
    what = {"a": "link", "link": "link", "button": "button", "input:submit": "button"}.get(e["kind"], e["kind"])
    return "CLICK", f"click the {what} {e['desc']}"


def action_key(e: dict) -> str:
    """Identity of an action across steps: element ids and typed values change, the rest does not."""
    return f"{e['kind']}|" + re.sub(r', value="[^"]*"', "", e["desc"])


def step_questions(obs: dict, blocked: set[str] = frozenset()) -> dict:
    """blocked: the previous action and every action already taken twice. Repeating actions was
    the main failure (the same query again and again, or search/type ping-pong), so they are not
    offered."""
    actions = {e["id"]: element_action(e)[1] for e in obs["elements"]
               if action_key(e) not in blocked and not ACCEPT_ALL.search(e["desc"])}
    actions["scroll"] = "scroll down to see more of the page"
    actions["back"] = "go back to the previous page"
    return {
        "done": {
            "type": "noul",
            "instructions": "Is the TASK already fully completed? Judge only from the visible page text "
                            "and the actions so far.",
            "criteria": {"true": "the page confirms that everything the task asks for is done",
                         "false": "at least one part of the task is still missing"},
        },
        "action": {
            "type": "choice",
            "instructions": "Which single action brings the browser one step closer to completing the TASK? "
                            "If the page already links to what the task needs, click it; otherwise a search "
                            "field is usually the shortest way. Use short search terms.",
            "criteria": actions,
        },
    }


# Overlay for recordings (--show): a caption bar with task and decision, and an outline on the
# chosen element. Only drawn between observation and action, never seen by the model.
OVERLAY_JS = r"""
([task, line, sub, id]) => {
  document.querySelectorAll('.sj-ov, .sj-mark').forEach(e => e.remove());
  const bar = document.createElement('div');
  bar.className = 'sj-ov';
  bar.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:2147483647;padding:14px 22px;' +
    'background:rgba(17,24,39,.93);color:#fff;font:16px/1.45 system-ui,sans-serif;box-shadow:0 -4px 18px rgba(0,0,0,.25)';
  const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
  bar.innerHTML = '<div style="opacity:.7;font-size:13px">Aufgabe: ' + esc(task) + '</div>' +
    '<div style="font-size:20px;font-weight:600;margin-top:2px">' + esc(line) + '</div>' +
    '<div style="opacity:.8;font-size:13px;margin-top:2px">' + esc(sub) + '</div>';
  document.body.appendChild(bar);
  const el = id && document.querySelector('[data-sj="' + id + '"]');
  if (el) {
    el.scrollIntoView({block: 'nearest'});
    // fixed boxes inside a CSS-zoomed page are scaled again: divide the rect by the zoom
    const z = parseFloat(document.documentElement.style.zoom) || 1;
    const b = el.getBoundingClientRect(), m = document.createElement('div');
    const r = {left: b.left / z, top: b.top / z, width: b.width / z, height: b.height / z};
    m.className = 'sj-mark';
    m.style.cssText = 'position:fixed;z-index:2147483646;pointer-events:none;border:3px solid #f59e0b;' +
      'border-radius:6px;box-shadow:0 0 0 4px rgba(245,158,11,.3);left:' + (r.left - 5) + 'px;top:' + (r.top - 5) +
      'px;width:' + (r.width + 4) + 'px;height:' + (r.height + 4) + 'px';
    document.body.appendChild(m);
  }
}
"""
OP_DE = {"CLICK": "Klicken", "TYPE": "Tippen", "TYPE_ENTER": "Tippen + Enter", "SELECT": "Auswählen",
         "SCROLL": "Scrollen", "BACK": "Zurück", "DONE": "Fertig ✓"}


GENERATE_SYSTEM = ("You are a browser agent. You see the task, the current page and the possible actions. "
                   "Decide the single next action.")
GENERATE_FORMAT = ('Reply with JSON only, in this format:\n'
                   '{"evaluation_previous_goal": "did the last action work?", "memory": "what matters so far", '
                   '"next_goal": "what to do next", "action": "<one id from POSSIBLE ACTIONS>", '
                   '"text": "<text to type, only for typing actions>", "option": "<dropdown entry, only for dropdowns>"}')


class BrowserAgent:
    """mode "snap": typed questions, probabilities read from the logits (snapjudge).
    mode "generate": the same model writes its decision as JSON, like browser-use and most LLM
    agents -- same page state, same action options, same repetition guard, for a fair comparison."""

    def __init__(self, engine, page, task: str, out_dir: Path, max_elements: int = 60, show_ms: int = 0,
                 mode: str = "snap"):
        self.engine, self.page, self.task, self.out = engine, page, task, out_dir
        self.mode = mode
        self.max_elements = max_elements
        self.show_ms = show_ms
        self.history: list[str] = []
        self.blocked: set[str] = set()
        self.done_count: dict[str, int] = {}

    def reject_cookies(self) -> bool:
        for label in REJECT_LABELS:
            btn = self.page.get_by_role("button", name=label, exact=True)
            try:
                if btn.count() and btn.first.is_visible():
                    btn.first.click(timeout=3000)
                    self.page.wait_for_load_state("domcontentloaded", timeout=8000)
                    self.page.wait_for_timeout(800)
                    return True
            except Exception:
                continue
        return False

    def observe(self) -> dict:
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass
        return self.page.evaluate(EXTRACT_JS, self.max_elements)

    def generate_text(self, state: str, element: dict) -> tuple[str, float]:
        """Ask the same model for the text to type into one field."""
        from mlx_vlm import generate

        t = time.perf_counter()
        messages = [
            {"role": "system", "content": "You fill in web forms for a browser agent. Reply with the exact "
                                          "text to type into the field, nothing else: no quotes, no explanation."},
            {"role": "user", "content": f"{state}\n\nFIELD: {element['id']} [{element['kind']}] {element['desc']}\n\n"
                                        "Text to type into this field:"},
        ]
        prompt = self.engine.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False)
        with self.engine.lock:
            res = generate(self.engine.model, self.engine.processor, prompt, max_tokens=48, temperature=0.0)
        text = res.text.strip().splitlines()[0].strip().strip('"').strip("'") if res.text.strip() else ""
        return text, (time.perf_counter() - t) * 1000

    def decide_generate(self, state: str, obs: dict) -> dict:
        """Classic LLM agent step: the model writes evaluation, memory, goal and action as JSON."""
        from mlx_vlm import generate

        actions = step_questions(obs, self.blocked)["action"]["criteria"]
        actions["done"] = "the task is complete: the page confirms everything the task asks for"
        listing = "\n".join(f"- {k}: {v}" for k, v in actions.items())
        messages = [
            {"role": "system", "content": GENERATE_SYSTEM},
            {"role": "user", "content": f"{state}\n\nPOSSIBLE ACTIONS:\n{listing}\n\n{GENERATE_FORMAT}"},
        ]
        prompt = self.engine.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False)
        with self.engine.lock:
            res = generate(self.engine.model, self.engine.processor, prompt, max_tokens=400, temperature=0.0)
        raw = res.text.strip()
        data = {}
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = {}
        if "action" not in data:
            m = re.search(r'"action"\s*:\s*"([^"]+)"', raw)
            data["action"] = m.group(1) if m else ""
        choice = str(data.get("action", "")).strip()
        if choice not in actions:  # e.g. "e3: type ..." or a stray label
            m = re.match(r"(e\d+|scroll|back|done)\b", choice)
            choice = m.group(1) if m and m.group(1) in actions else "invalid"
        return {"choice": choice, "text": data.get("text") or None, "option": data.get("option") or None,
                "raw": raw[:600], "prompt_tokens": res.prompt_tokens, "generation_tokens": res.generation_tokens}

    def pick_option(self, state: str, element: dict) -> tuple[str, dict]:
        """For SELECT: a second typed question over the dropdown's own entries."""
        opts = {f"o{i}": text or "(empty)" for i, text in enumerate(element["options"] or [])}
        res = self.engine.system_one(state, {"option": {
            "type": "choice",
            "instructions": f"Which entry of dropdown {element['id']} should be chosen for the TASK?",
            "criteria": opts,
        }}, layout="state_first")
        ans = res["answers"]["option"]
        return element["options"][int(ans["choice"][1:])], ans

    def act(self, op: str, element: dict | None, text: str | None, option: str | None) -> str:
        loc = self.page.locator(f'[data-sj="{element["id"]}"]') if element else None
        if op == "CLICK":
            loc.click(timeout=5000)
        elif op in ("TYPE", "TYPE_ENTER"):
            # Click, clear and type key by key: autocomplete widgets (airports, cities) only open
            # their suggestion list on real key events, not on a value set with fill().
            loc.click(timeout=5000)
            self.page.keyboard.press("ControlOrMeta+a")
            self.page.keyboard.press("Backspace")
            self.page.keyboard.type(text or "", delay=25)
            if op == "TYPE_ENTER":
                self.page.keyboard.press("Enter")
            else:
                self.page.wait_for_timeout(900)  # let suggestions appear before the next observation
        elif op == "SELECT":
            loc.select_option(label=option, timeout=5000)
        elif op == "SCROLL":
            self.page.mouse.wheel(0, 700)
        elif op == "BACK":
            self.page.go_back(timeout=5000)
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass
        return "ok"

    def run(self, max_steps: int = 15) -> dict:
        trace = self.out / "trace.jsonl"
        t_run = time.perf_counter()
        status, repeats, last = "max_steps", 0, None
        if self.reject_cookies():
            self.history.append("0. cookie banner: rejected non-essential cookies (automatic)")
        for step in range(1, max_steps + 1):
            rec: dict = {"step": step}
            t = time.perf_counter()
            obs = self.observe()
            rec["observe_ms"] = round((time.perf_counter() - t) * 1000)
            rec["url"], rec["n_elements"] = obs["url"], len(obs["elements"])
            state = page_state(self.task, obs, self.history)

            t = time.perf_counter()
            gen = None
            if self.mode == "generate":
                gen = self.decide_generate(state, obs)
                rec["decide_ms"] = round((time.perf_counter() - t) * 1000)
                choice = gen["choice"]
                rec["p_done"] = rec["action_p"] = None
                rec.update({k: gen[k] for k in ("raw", "prompt_tokens", "generation_tokens")})
                is_done = choice == "done"
            else:
                res = self.engine.system_one(state, step_questions(obs, self.blocked), layout="state_first")
                rec["decide_ms"] = round((time.perf_counter() - t) * 1000)
                p_done = res["answers"]["done"]["noul"]
                act_ans = res["answers"]["action"]
                choice = act_ans["choice"]
                rec["p_done"] = p_done
                rec["action_p"] = act_ans["probabilities"][choice]
                rec["action_top3"] = dict(sorted(act_ans["probabilities"].items(), key=lambda kv: -kv[1])[:3])
                is_done = p_done >= DONE_THRESHOLD
            element = None
            if is_done:
                op = "DONE"
            elif choice == "invalid":
                op = "INVALID"
            elif choice in ("scroll", "back"):
                op = choice.upper()
            else:
                element = next(e for e in obs["elements"] if e["id"] == choice)
                op = element_action(element)[0]
                rec["target"] = f"{element['id']} [{element['kind']}] {element['desc']}"
            rec["operation"] = op

            text = option = None
            if gen and op in ("TYPE", "TYPE_ENTER"):
                text = str(gen["text"] or "")
                rec["text"] = text
            elif gen and op == "SELECT":
                option = str(gen["option"] or "")
                rec["option"] = option
            elif op in ("TYPE", "TYPE_ENTER") and element:
                text, rec["generate_ms"] = self.generate_text(state, element)
                rec["generate_ms"] = round(rec["generate_ms"])
                rec["text"] = text
            if op == "SELECT" and element and element.get("options"):
                option, _ = self.pick_option(state, element)
                rec["option"] = option

            if self.show_ms:
                self._show(step, op, element, text, option, rec)
            if op == "DONE":
                status = "done"
                self._log(trace, rec)
                break
            if op == "INVALID":  # unparseable answer: counts as a step, nothing happens
                rec["result"] = "invalid answer"
                self.history.append(f"{step}. (invalid answer) -> nothing happened")
                self._log(trace, rec)
                continue

            t = time.perf_counter()
            try:
                rec["result"] = self.act(op, element, text, option)
            except Exception as exc:  # the next observation shows the model what happened
                rec["result"] = f"error: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
            rec["act_ms"] = round((time.perf_counter() - t) * 1000)
            if element:
                key = action_key(element)
                self.done_count[key] = self.done_count.get(key, 0) + 1
                self.blocked = {key} | {k for k, n in self.done_count.items() if n >= 2}
            else:
                self.blocked = {k for k, n in self.done_count.items() if n >= 2}

            desc = f"{op} {element['id']} ({element['desc'][:50]})" if element else op
            if text is not None:
                desc += f' text="{text}"'
            if option is not None:
                desc += f' option="{option}"'
            self.history.append(f"{step}. {desc} -> {rec['result']}")
            self._log(trace, rec)

            key = (op, element["desc"] if element else None, text, option)
            repeats = repeats + 1 if key == last else 0
            last = key
            if repeats >= 2:
                status = "stuck"
                break

        self.page.screenshot(path=str(self.out / "final.png"))
        summary = {"task": self.task, "status": status, "steps": len(self.history) + (status == "done"),
                   "seconds": round(time.perf_counter() - t_run, 2), "final_url": self.page.url,
                   "model": self.engine.name, "history": self.history}
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary

    def _show(self, step, op, element, text, option, rec):
        what = ""
        if element:
            m = re.search(r'"([^"]+)"', element["desc"])
            what = f" „{m.group(1)}“" if m else f" {element['kind']}"
        if text is not None:
            what += f" → „{text}“"
        if option is not None:
            what += f" → {option}"
        p = rec["p_done"] if op == "DONE" else rec["action_p"]
        how = f"p = {p:.2f}" if p is not None else f"{rec.get('generation_tokens')} Token geschrieben"
        sub = (f"Qwen3.5-4B lokal · Entscheidung in {rec['decide_ms']} ms · {how}"
               + (f" · Text in {rec['generate_ms']} ms" if "generate_ms" in rec else ""))
        try:
            self.page.evaluate(OVERLAY_JS, [self.task, f"Schritt {step}: {OP_DE.get(op, op)}{what}", sub,
                                            element["id"] if element else None])
            self.page.wait_for_timeout(self.show_ms * (2 if op == "DONE" else 1))
            self.page.evaluate("document.querySelectorAll('.sj-ov, .sj-mark').forEach(e => e.remove())")
        except Exception:
            pass

    @staticmethod
    def _log(path: Path, rec: dict):
        with path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        extra = f" {rec['target']}" if "target" in rec else ""
        if "text" in rec:
            extra += f' text="{rec["text"]}"'
        if "option" in rec:
            extra += f' option="{rec["option"]}"'
        print(f"[{rec['step']:2d}] {rec['operation']:10s}{extra[:110]}  "
              f"(decide {rec['decide_ms']} ms"
              + (f", p_action {rec['action_p']:.2f}, p_done {rec['p_done']:.2f})" if rec.get("action_p") is not None
                 else f", {rec.get('generation_tokens')} tokens written)")
              + 
              f" -> {rec.get('result', '')}",
              flush=True)


def main():
    from playwright.sync_api import sync_playwright

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--task")
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--calibration")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--max-elements", type=int, default=60)
    ap.add_argument("--headful", action="store_true", help="show the browser window")
    ap.add_argument("--channel", help="installed browser to use instead of Playwright's own, e.g. chrome")
    ap.add_argument("--dry-run", action="store_true", help="print the page state and questions, no model")
    ap.add_argument("--mode", choices=["snap", "generate"], default="snap",
                    help="snap: typed decisions from the logits; generate: classic agent writing JSON")
    ap.add_argument("--show", type=int, default=0, metavar="MS", help="show each decision on the page for MS ms")
    ap.add_argument("--video", action="store_true", help="record a video of the run into the output directory")
    ap.add_argument("--viewport", default="1280x900", help="WIDTHxHEIGHT, e.g. 960x540 for a compact recording")
    ap.add_argument("--zoom", type=float, default=1.0, help="CSS zoom of every page, for readable recordings")
    ap.add_argument("--out", help="output directory (default results/browser/<timestamp>)")
    args = ap.parse_args()

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=not args.headful, channel=args.channel)
        except Exception:
            if args.channel:
                raise
            # Playwright's bundled browser not downloaded: fall back to an installed Chrome
            browser = pw.chromium.launch(headless=not args.headful, channel="chrome")
        out = Path(args.out) if args.out else ROOT / "results" / "browser" / datetime.now().strftime("%Y%m%d-%H%M%S")
        w, h = (int(x) for x in args.viewport.split("x"))
        ctx_args = {"viewport": {"width": w, "height": h}, "locale": "de-DE"}
        if args.video and not args.dry_run:
            ctx_args["record_video_dir"] = str(out / "video")
            ctx_args["record_video_size"] = ctx_args["viewport"]
        context = browser.new_context(**ctx_args)
        if args.zoom != 1.0:
            context.add_init_script(f"document.addEventListener('DOMContentLoaded', () => "
                                    f"{{ document.documentElement.style.zoom = '{args.zoom}'; }});")
        if args.model and args.video:
            from snapjudge.engine import Engine  # load before the page opens: no idle seconds on video
            engine = Engine(args.model, adapter_path=args.adapter, calibration_path=args.calibration)
        page = context.new_page()
        page.goto(args.url, timeout=20000)

        if args.dry_run:
            agent = BrowserAgent(None, page, args.task or "(no task)", Path("."), args.max_elements)
            print("cookie banner rejected:", agent.reject_cookies())
            obs = agent.observe()
            state = page_state(agent.task, obs, [])
            print(state)
            print(f"\n--- {len(state)} characters, {len(obs['elements'])} elements ---")
            print(json.dumps(step_questions(obs)["action"]["criteria"], indent=2, ensure_ascii=False))
            browser.close()
            return

        if not (args.model and args.task):
            ap.error("--model and --task are required unless --dry-run")
        if not args.video:
            from snapjudge.engine import Engine

            engine = Engine(args.model, adapter_path=args.adapter, calibration_path=args.calibration)
        out.mkdir(parents=True, exist_ok=True)
        summary = BrowserAgent(engine, page, args.task, out, args.max_elements, args.show, args.mode).run(args.max_steps)
        context.close()  # writes the video
        browser.close()
    print(f"\n{summary['status'].upper()} after {summary['steps']} steps, {summary['seconds']} s  -> {out}")


if __name__ == "__main__":
    main()
