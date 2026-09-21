"""Typed decisions from a local Qwen model on MLX, without generating text.

Per request:
1. Prefill the prompt head once and keep a cache snapshot (LRU across requests). The head is
   system + question (the same question repeats across many requests, so each request only
   pays for its state) or system + state (a long state evaluated against several questions).
2. Tokenize the allowed labels of each question into a prefix tree.
3. One row per branching node in the tree (rest of the prompt + path); all rows of all
   questions run as one batch from the snapshot.
4. At each branching node, softmax over the allowed tokens only (temperature T); the product
   along the path is the probability of the option.

Built for the Qwen 3.5 family (qwen3_5 / qwen3_5_moe, which includes Qwen3.6 and Qwen3.8).
Their hybrid attention keeps recurrent state that cannot be trimmed, hence snapshot/restore.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.models.cache import ArraysCache, KVCache, make_prompt_cache

from .prompts import SYSTEM_PROMPT, question_block, question_header, render_value, state_block

_SPLIT = "⁣<<SO-SPLIT>>⁣"
_SUPPORTED = {"qwen3_5", "qwen3_5_moe"}
LONG_STATE_CHARS = 2000  # above this, with several questions, prefill the state only once
SHARED_SUFFIX_TOKENS = 256  # question part this long, with 2+ branches: prefill it once


@dataclass
class _Node:
    children: dict = field(default_factory=dict)  # token id -> _Node
    options: list = field(default_factory=list)  # options whose label ends here


@dataclass
class _Question:
    qid: str
    qtype: str
    options: list
    criteria: object
    suffix: list  # tokens after the head
    root: _Node
    branches: list = field(default_factory=list)  # (node, path_tokens)


class Engine:
    def __init__(
        self,
        model_path: str,
        name: str | None = None,
        calibration_path: str | None = None,
        prefix_slots: int = 16,
        row_budget_gb: float = 4.0,
        max_rows: int = 48,
        prefill_chunk: int = 2048,
        cache_limit_gb: float | None = None,
        adapter_path: str | None = None,
        fuse_adapter: bool = True,
    ):
        self.model_path = model_path
        self.name = name or Path(model_path).name + (f"+{Path(adapter_path).name}" if adapter_path else "")
        # MLX keeps freed buffers in its own cache. After large batches that can pin many GB,
        # which matters when other model servers share the machine. Cap it (SO_CACHE_LIMIT_GB).
        limit = cache_limit_gb if cache_limit_gb is not None else float(os.environ.get("SO_CACHE_LIMIT_GB", 4))
        mx.set_cache_limit(int(limit * 1024**3))
        self.model, self.processor = load(model_path, adapter_path=adapter_path)
        if adapter_path and fuse_adapter:
            self._fuse_lora()
        self.lm = self.model.language_model
        mt = getattr(self.model.config, "model_type", "?")
        if mt not in _SUPPORTED:
            raise ValueError(f"Model type {mt} not supported (expected one of {sorted(_SUPPORTED)})")
        self.tok = getattr(self.processor, "tokenizer", self.processor)
        self.end_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.temps = self._load_temps(calibration_path)
        self.prefix_slots = prefix_slots
        self.row_budget = int(row_budget_gb * 1024**3)
        self.max_rows = max_rows
        self.prefill_chunk = prefill_chunk
        self._prefixes: OrderedDict[str, list] = OrderedDict()
        self._base_ids = self._encode(self._template(_SPLIT)[0])  # chat template up to the user content
        self._base_snap = None
        self.lock = threading.Lock()

    def _fuse_lora(self):
        """Fold LoRA deltas into the base weights. The fused layers stay in bf16: re-quantizing them
        to 4 bit erased most of the fine-tuning (typed-decisions accuracy 0.775 -> 0.606)."""
        from mlx.utils import tree_unflatten

        fused = [(n, m.fuse(dequantize=True)) for n, m in self.model.named_modules() if hasattr(m, "fuse") and hasattr(m, "lora_a")]
        if fused:
            self.model.update_modules(tree_unflatten(fused))
        # Materialize the fused weights here: left lazy, they are computed on first use, and a
        # server answers from a worker thread, where MLX has no stream for this thread's graph.
        mx.eval(self.model.parameters())
        self.model.eval()

    # ------------------------------------------------------------------ calibration
    def _load_temps(self, path: str | None) -> dict:
        path = path or os.environ.get("SO_CALIBRATION")
        if not path or not Path(path).exists():
            return {}
        data = json.loads(Path(path).read_text())
        return data.get(self.name, {})

    # ------------------------------------------------------------------ tokenization
    def _encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def _render(self, state, question_text: str, layout: str = "state_first", header: str = "") -> tuple[str, str]:
        """Split the prompt into head (cached) and rest. state_first puts the state in the head
        (many questions about one state); question_first puts the question in the head (the same
        question about many states); header puts a list of all questions plus the state in the
        head, so the state is read with the questions in view and still prefilled only once."""
        if layout == "state_first":
            return self._template(state_block(state) + _SPLIT + question_text)
        if layout == "header":
            return self._template(header + state_block(state) + _SPLIT + question_text)
        return self._template(f"{question_text}\n\n<state>\n{_SPLIT}{render_value(state)}\n</state>")

    def _template(self, content: str) -> tuple[str, str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        rendered = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        head, tail = rendered.split(_SPLIT)
        return head, tail

    def _label_tokens(self, tail: str, tail_ids: list[int], label: str) -> list[int]:
        ids = self._encode(tail + label)
        if ids[: len(tail_ids)] != tail_ids:
            raise RuntimeError(f"Label {label!r} merges with the end of the prompt when tokenized")
        return ids[len(tail_ids):]

    # ------------------------------------------------------------------ label prefix tree
    def _build_trie(self, paths: list[tuple[str, list[int]]]) -> _Node:
        root = _Node()
        for option, tokens in paths:
            node = root
            for t in tokens:
                node = node.children.setdefault(t, _Node())
            if option not in node.options:
                node.options.append(option)
        return root

    def _outgoing(self, node: _Node) -> list[int]:
        out = list(node.children)
        if node.options and node.children:
            out.append(self.end_id)  # a label ends here while a longer one continues
        return out

    def _collect_branches(self, node: _Node, path: list[int], acc: list):
        if len(self._outgoing(node)) >= 2:
            acc.append((node, path))
        for tok, child in node.children.items():
            self._collect_branches(child, path + [tok], acc)

    def _option_probs(self, q: _Question, logits_at: dict, temp: float) -> dict:
        probs = {o: 0.0 for o in q.options}

        def walk(node: _Node, p: float):
            out = self._outgoing(node)
            if len(out) >= 2:
                z = logits_at[id(node)][out] / temp
                z = np.exp(z - z.max())
                branch = z / z.sum()
            else:
                branch = np.ones(len(out))
            for tok, bp in zip(out, branch):
                if tok == self.end_id and not node.children.get(tok):
                    for o in node.options:
                        probs[o] += p * bp / len(node.options)
                else:
                    walk(node.children[tok], p * bp)
            if node.options and not node.children:
                for o in node.options:
                    probs[o] += p / len(node.options)

        walk(q.root, 1.0)
        total = sum(probs.values()) or 1.0
        return {o: v / total for o, v in probs.items()}

    # ------------------------------------------------------------------ MLX forward
    def _forward(self, tokens: mx.array, cache, offset: int, gather: list[int] | None):
        B, L = tokens.shape
        pos = mx.arange(offset, offset + L, dtype=mx.int32)
        pos = mx.broadcast_to(pos[None, None, :], (3, B, L))
        out = self.lm(tokens, cache=cache, position_ids=pos, return_hidden=True, skip_logits=True)
        if gather is None:
            return None
        hidden = out.hidden_states[-1]
        # LM head in float32: bf16 logits around 20 are quantized in steps of 0.125.
        h = hidden[mx.arange(B), mx.array(gather)][:, None, :].astype(mx.float32)
        if self.lm.args.tie_word_embeddings:
            logits = self.lm.model.embed_tokens.as_linear(h)
        else:
            logits = self.lm.lm_head(h)
        return logits[:, 0, :].astype(mx.float32)

    @staticmethod
    def _cache_arrays(cache) -> list:
        arrays = []
        for c in cache:
            if isinstance(c, KVCache):
                if c.keys is not None:
                    arrays += [c.keys, c.values]
            else:
                arrays += [a for a in c.cache if a is not None]
        return arrays

    @staticmethod
    def _snapshot(cache) -> list:
        snap = []
        for c in cache:
            if isinstance(c, KVCache):
                k, v = c.state
                snap.append(("kv", k, v))
            elif isinstance(c, ArraysCache):
                snap.append(("arr", list(c.cache)))
            else:
                raise TypeError(f"Cache type {type(c).__name__} not supported")
        mx.eval([a for s in snap for a in (s[1:] if s[0] == "kv" else s[1]) if a is not None])
        return snap

    def _restore(self, snap: list, batch: int):
        cache = make_prompt_cache(self.lm)
        for c, s in zip(cache, snap):
            if s[0] == "kv":
                k, v = s[1], s[2]
                if batch > 1:
                    k, v = mx.repeat(k, batch, axis=0), mx.repeat(v, batch, axis=0)
                c.state = (k, v)
            else:
                c.cache = [
                    None if a is None else (mx.repeat(a, batch, axis=0) if batch > 1 else a)
                    for a in s[1]
                ]
        return cache

    @staticmethod
    def _snapshot_bytes(snap: list) -> int:
        n = 0
        for s in snap:
            arrays = s[1:] if s[0] == "kv" else s[1]
            n += sum(a.nbytes for a in arrays if a is not None)
        return n

    def _prefill(self, ids: list[int], cache, start: int = 0):
        for i in range(start, len(ids), self.prefill_chunk):
            chunk = mx.array(ids[i : i + self.prefill_chunk])[None]
            self._forward(chunk, cache, offset=i, gather=None)
            mx.eval(self._cache_arrays(cache))

    def _prefix(self, head_ids: list[int]) -> tuple[list, bool]:
        key = hashlib.sha1(np.asarray(head_ids, dtype=np.int32).tobytes()).hexdigest()
        if key in self._prefixes:
            self._prefixes.move_to_end(key)
            return self._prefixes[key], True
        # The system prompt is identical for every request: prefill it once, then extend
        if self._base_snap is None:
            cache = make_prompt_cache(self.lm)
            self._prefill(self._base_ids, cache)
            self._base_snap = self._snapshot(cache)
        n_base = len(self._base_ids)
        if head_ids[:n_base] == self._base_ids and len(head_ids) > n_base:
            cache, start = self._restore(self._base_snap, 1), n_base
        else:
            cache, start = make_prompt_cache(self.lm), 0
        self._prefill(head_ids, cache, start)
        snap = self._snapshot(cache)
        self._prefixes[key] = snap
        while len(self._prefixes) > self.prefix_slots:
            self._prefixes.popitem(last=False)
        return snap, False

    def _run_rows(self, snap: list, offset: int, rows: list[list[int]]) -> list[np.ndarray]:
        per_row = max(1, self._snapshot_bytes(snap))
        batch_size = max(1, min(self.max_rows, self.row_budget // per_row))
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
        results: list = [None] * len(rows)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            width = max(len(rows[i]) for i in idx)
            padded = [rows[i] + [self.end_id] * (width - len(rows[i])) for i in idx]
            cache = self._restore(snap, len(idx))
            logits = self._forward(
                mx.array(padded, dtype=mx.int32), cache, offset, gather=[len(rows[i]) - 1 for i in idx]
            )
            mx.eval(logits)
            arr = np.array(logits)
            for j, i in enumerate(idx):
                results[i] = arr[j]
            del cache, logits
        return results

    # ------------------------------------------------------------------ public API
    def system_one(self, state, questions: dict, debug: bool = False, layout: str = "auto") -> dict:
        """Answer typed questions about a state. layout: auto | state_first | question_first | header.

        question_first caches the question (it repeats across requests) and computes only the
        state per request; measured faster and no less accurate. state_first pays off when a long
        state is evaluated against several questions, because the state is prefilled only once.
        header lists all questions before the state and then asks each one: the state is still
        prefilled once, but read with the questions in view (on long invoices it recovered about
        half of question_first's accuracy gain at state_first's cost)."""
        t0 = time.perf_counter()
        if layout == "auto":
            long_state = len(render_value(state)) > LONG_STATE_CHARS
            layout = "state_first" if long_state and len(questions) > 1 else "question_first"
        header = question_header(questions) if layout == "header" else ""
        heads: dict[tuple, list[int]] = {}  # head tokens -> indices of questions
        parsed: list[_Question] = []
        for qid, q in questions.items():
            text, options, surfaces = question_block(q)
            head, tail = self._render(state, text, layout, header)
            head_ids = self._encode(head)
            full = self._encode(head + tail)
            if full[: len(head_ids)] != head_ids:
                raise RuntimeError("Head and rest of the prompt merge when tokenized")
            tail_ids = self._encode(tail)
            paths = [
                (opt, self._label_tokens(tail, tail_ids, form))
                for opt, forms in surfaces.items()
                for form in forms
            ]
            pq = _Question(qid, q["type"], options, q.get("criteria"), full[len(head_ids):], self._build_trie(paths))
            self._collect_branches(pq.root, [], pq.branches)
            heads.setdefault(tuple(head_ids), []).append(len(parsed))
            parsed.append(pq)

        t1 = time.perf_counter()
        prefill_s, hits, n_rows = 0.0, [], 0
        logits_by_q: list[dict] = [dict() for _ in parsed]
        root_logits: list = [None] * len(parsed)
        for head_ids, members in heads.items():
            ta = time.perf_counter()
            snap, hit = self._prefix(list(head_ids))
            prefill_s += time.perf_counter() - ta
            hits.append(hit)
            rows, owners = [], []
            for qi in members:
                pq = parsed[qi]
                if len(pq.branches) >= 2 and len(pq.suffix) >= SHARED_SUFFIX_TOKENS:
                    # Many branches under a long question (e.g. dozens of element labels): prefill
                    # the question once and run only the label paths, instead of the whole question
                    # once per branch.
                    ta = time.perf_counter()
                    cache = self._restore(snap, 1)
                    self._prefill(list(head_ids) + pq.suffix[:-1], cache, start=len(head_ids))
                    q_snap = self._snapshot(cache)
                    prefill_s += time.perf_counter() - ta
                    q_rows = [[pq.suffix[-1]] + path for _, path in pq.branches]
                    n_rows += len(q_rows)
                    offset = len(head_ids) + len(pq.suffix) - 1
                    for (node, path), lg in zip(pq.branches, self._run_rows(q_snap, offset, q_rows)):
                        logits_by_q[qi][id(node)] = lg
                        if not path:
                            root_logits[qi] = lg
                    del cache, q_snap
                    continue
                for node, path in pq.branches:
                    rows.append(pq.suffix + path)
                    owners.append((qi, node, not path))
            n_rows += len(rows)
            for (qi, node, is_root), lg in zip(owners, self._run_rows(snap, len(head_ids), rows) if rows else []):
                logits_by_q[qi][id(node)] = lg
                if is_root:
                    root_logits[qi] = lg
        t3 = time.perf_counter()
        t2 = t1 + prefill_s

        answers, dbg = {}, {}
        for qi, pq in enumerate(parsed):
            temp = float(self.temps.get(pq.qtype, 1.0))
            probs = self._option_probs(pq, logits_by_q[qi], temp)
            answers[pq.qid] = self._answer(pq, probs)
            if debug:
                dbg[pq.qid] = self._debug_info(pq, probs, logits_by_q[qi], root_logits[qi], temp)

        result = {
            "model": self.name,
            "answers": answers,
            "usage": {
                "input_tokens": sum(len(h) for h in heads) + sum(len(pq.suffix) for pq in parsed),
                "output_tokens": n_rows,
            },
        }
        if debug:
            result["debug"] = {
                "questions": dbg,
                "layout": layout,
                "prefix_tokens": sum(len(h) for h in heads),
                "prefix_cache_hit": all(hits),
                "rows": n_rows,
                "timing_ms": {
                    "tokenize": round((t1 - t0) * 1000, 1),
                    "prefill": round((t2 - t1) * 1000, 1),
                    "questions": round((t3 - t2) * 1000, 1),
                    "total": round((time.perf_counter() - t0) * 1000, 1),
                },
            }
        return result

    @staticmethod
    def _confidence(probs: list[float]) -> float:
        """1 - normalized entropy; matches the example in the TypeSafe quickstart."""
        n = len(probs)
        if n < 2:
            return 1.0
        h = -sum(p * math.log(p) for p in probs if p > 0)
        return max(0.0, 1.0 - h / math.log(n))

    def _answer(self, q: _Question, probs: dict) -> dict:
        r = lambda x: round(float(x), 4)
        if q.qtype == "noul":
            return {"type": "noul", "noul": r(probs["yes"])}
        if q.qtype == "choice":
            best = max(q.options, key=lambda o: probs[o])
            return {
                "type": "choice",
                "choice": best,
                "probabilities": {o: r(probs[o]) for o in q.options},
                "confidence": r(self._confidence(list(probs.values()))),
            }
        levels = q.options
        return {
            "type": "score",
            "score": r(sum(int(lvl) * probs[lvl] for lvl in levels)),
            "legend": {lvl: str(desc) for lvl, desc in zip(levels, q.criteria)},
            "probabilities": {lvl: r(probs[lvl]) for lvl in levels},
            "confidence": r(self._confidence(list(probs.values()))),
        }

    def _debug_info(self, q: _Question, probs, logits_at, root_lg, temp) -> dict:
        info = {"temperature": temp, "branches": len(q.branches)}
        if temp != 1.0:
            info["raw_probabilities"] = {o: round(v, 4) for o, v in self._option_probs(q, logits_at, 1.0).items()}
        if root_lg is not None:
            z = root_lg - root_lg.max()
            full = np.exp(z) / np.exp(z).sum()
            allowed = self._outgoing(q.root)
            info["coverage"] = round(float(full[allowed].sum()), 4)
            top = np.argsort(-full)[:5]
            info["top_tokens"] = [[self.tok.decode([int(t)]), round(float(full[t]), 4)] for t in top]
        return info
