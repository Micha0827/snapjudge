"""Run the browser agent over a task list with one loaded model and score the outcome.

    python agent/run_suite.py --model <mlx model> [--adapter dir] [--tasks agent/aufgaben.json] [--tag name]

A task counts as solved when the agent says DONE and the final page matches `expect_url`
(substring of the URL) or `expect_text` (substring of the page text). Relative URLs point into
agent/.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

AGENT = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT))
sys.path.insert(0, str(AGENT.parent))

from browser_agent import BrowserAgent  # noqa: E402


def main():
    from playwright.sync_api import sync_playwright
    from snapjudge.engine import Engine

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--calibration")
    ap.add_argument("--tasks", default=str(AGENT / "aufgaben.json"))
    ap.add_argument("--only", help="comma-separated task ids")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--tag")
    ap.add_argument("--mode", choices=["snap", "generate"], default="snap")
    args = ap.parse_args()

    tasks = json.loads(Path(args.tasks).read_text())
    if args.only:
        keep = set(args.only.split(","))
        tasks = [t for t in tasks if t["id"] in keep]
    engine = Engine(args.model, adapter_path=args.adapter, calibration_path=args.calibration)
    tag = args.tag or engine.name
    base = AGENT.parent / "results" / "browser" / f"suite-{tag}"

    rows = []
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception:
            browser = pw.chromium.launch(channel="chrome")
        for t in tasks:
            page = browser.new_page(viewport={"width": 1280, "height": 900}, locale="de-DE")
            url = t["url"] if "://" in t["url"] else (AGENT / t["url"]).resolve().as_uri()
            page.goto(url, timeout=20000)
            out = base / t["id"]
            out.mkdir(parents=True, exist_ok=True)
            (out / "trace.jsonl").unlink(missing_ok=True)
            print(f"\n=== {t['id']}: {t['task']}", flush=True)
            summary = BrowserAgent(engine, page, t["task"], out, mode=args.mode).run(args.max_steps)
            text = page.evaluate("document.body ? document.body.innerText : ''")
            hit = (t.get("expect_url", "\0") in page.url) or (t.get("expect_text", "\0") in text)
            trace = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]
            decide = [r["decide_ms"] for r in trace]
            tokens = [r["generation_tokens"] for r in trace if "generation_tokens" in r]
            rows.append({"id": t["id"], "status": summary["status"], "reached": hit,
                         "solved": hit and summary["status"] == "done", "steps": summary["steps"],
                         "seconds": summary["seconds"], "decide_ms_median": statistics.median(decide) if decide else None,
                         "tokens_per_step": round(statistics.mean(tokens)) if tokens else None})
            r = rows[-1]
            print(f"--> {'GELÖST' if r['solved'] else 'nicht gelöst'} ({r['status']}, Ziel erreicht: {hit}) "
                  f"{r['steps']} Schritte, {r['seconds']} s, Entscheidung Median {r['decide_ms_median']} ms", flush=True)
            page.close()
        browser.close()

    solved = sum(r["solved"] for r in rows)
    result = {"model": engine.name, "mode": args.mode, "solved": solved, "n": len(rows), "tasks": rows}
    (base / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n{tag}: {solved}/{len(rows)} gelöst")
    for r in rows:
        print(f"  {r['id']:22s} {'✓' if r['solved'] else '✗'} {r['status']:9s} {r['steps']:2d} Schritte "
              f"{r['seconds']:6.1f} s  Median {r['decide_ms_median']} ms  Token/Schritt {r['tokens_per_step']}")


if __name__ == "__main__":
    main()
