"""Run the LocalLLaMA/typed-decisions benchmark (Apache-2.0) against this engine.

400 test cases x 5 typed questions over four workflows. The dataset card reports Jev 1.13.0
at 0.727 accuracy (measured through the TypeSafe API) against a 0.735 teacher ceiling.

    python eval/extern/typed_decisions.py <test.parquet> --local <model> [--adapter <dir>] [--tag name]

Download the split first, e.g.
    https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/main/all/test-00000-of-00001.parquet

Metrics follow the card as closely as the card defines them: accuracy of the argmax against the
gold label, KL(gold || prediction), total variation, Brier against the gold distribution, ECE of
the top-label confidence, and the absolute error of the expected score for score questions.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EPS = 1e-6


def predicted_distribution(ans: dict) -> dict:
    if ans["type"] == "noul":
        return {"true": ans["noul"], "false": 1 - ans["noul"]}
    return ans["probabilities"]


def decision_metrics(pred: dict, gold: dict, qtype: str) -> dict:
    keys = sorted(set(gold["probabilities"]) | set(pred))
    g = {k: float(gold["probabilities"].get(k, 0.0)) for k in keys}
    p = {k: max(float(pred.get(k, 0.0)), EPS) for k in keys}
    z = sum(p.values())
    p = {k: v / z for k, v in p.items()}
    top = max(p, key=p.get)
    out = {
        "correct": top == str(gold["label"]),
        "conf": p[top],
        "kl": sum(gv * math.log(max(gv, EPS) / p[k]) for k, gv in g.items() if gv > 0),
        "tv": 0.5 * sum(abs(p[k] - g[k]) for k in keys),
        "brier": sum((p[k] - g[k]) ** 2 for k in keys),
    }
    if qtype == "score":
        exp_p = sum(int(k) * v for k, v in p.items())
        out["score_ae"] = abs(exp_p - float(gold.get("score", sum(int(k) * v for k, v in g.items()))))
    return out


def ece(rows: list[dict], bins: int = 10) -> float:
    total, n = 0.0, len(rows)
    for b in range(bins):
        sel = [r for r in rows if b / bins < r["conf"] <= (b + 1) / bins or (b == 0 and r["conf"] == 0)]
        if sel:
            total += len(sel) / n * abs(sum(r["correct"] for r in sel) / len(sel) - sum(r["conf"] for r in sel) / len(sel))
    return total


def summarize(rows: list[dict]) -> dict:
    scores = [r["score_ae"] for r in rows if "score_ae" in r]
    return {
        "n": len(rows),
        "acc": sum(r["correct"] for r in rows) / len(rows),
        "kl": statistics.mean(r["kl"] for r in rows),
        "tv": statistics.mean(r["tv"] for r in rows),
        "brier": statistics.mean(r["brier"] for r in rows),
        "ece": ece(rows),
        "score_mae": statistics.mean(scores) if scores else None,
    }


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--local", required=True, help="model path or Hugging Face id")
    ap.add_argument("--adapter", help="LoRA adapter directory")
    ap.add_argument("--no-fuse", action="store_true", help="keep the adapter as separate LoRA layers")
    ap.add_argument("--calibration", help="per-type temperatures, JSON {model_name: {type: T}}")
    ap.add_argument("--tag")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    from snapjudge.engine import Engine

    engine = Engine(args.local, adapter_path=args.adapter, fuse_adapter=not args.no_fuse,
                    calibration_path=args.calibration)
    df = pd.read_parquet(args.parquet)
    if args.limit:
        df = df.head(args.limit)
    first = df.iloc[0]
    engine.system_one(json.loads(first["state"]), json.loads(first["questions"]))  # warm-up

    rows, preds, ms = [], [], []
    for _, r in df.iterrows():
        state, questions, gold = json.loads(r["state"]), json.loads(r["questions"]), json.loads(r["gold"])
        t = time.perf_counter()
        res = engine.system_one(state, questions)
        ms.append((time.perf_counter() - t) * 1000)
        for qid, q in questions.items():
            m = decision_metrics(predicted_distribution(res["answers"][qid]), gold[qid], q["type"])
            m.update({"workflow": r["workflow"], "type": q["type"], "qid": qid})
            rows.append(m)
        preds.append({"id": r["id"], "answers": res["answers"]})

    tag = args.tag or engine.name
    out_dir = ROOT / "results" / "typed-decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
    summary = {"model": engine.name, "adapter": args.adapter, "ms_per_case_median": statistics.median(ms),
               "overall": summarize(rows),
               "per_workflow": {w: summarize([x for x in rows if x["workflow"] == w]) for w in sorted({x["workflow"] for x in rows})},
               "per_type": {t: summarize([x for x in rows if x["type"] == t]) for t in sorted({x["type"] for x in rows})}}
    (out_dir / f"{tag}.summary.json").write_text(json.dumps(summary, indent=2))

    o = summary["overall"]
    print(f"{tag}: acc {o['acc']:.3f} | KL {o['kl']:.3f} | TV {o['tv']:.3f} | Brier {o['brier']:.3f} | "
          f"ECE {o['ece']:.3f} | score MAE {o['score_mae']:.3f} | {summary['ms_per_case_median']:.0f} ms/case "
          f"(Jev 1.13.0: acc 0.727, KL 1.442, TV 0.251, Brier 0.148, ECE 0.144, score MAE 0.391, 710 ms/case)")
    for w, s in summary["per_workflow"].items():
        print(f"  {w:28s} acc {s['acc']:.3f}  KL {s['kl']:.3f}")
    for t, s in summary["per_type"].items():
        print(f"  {t:8s} acc {s['acc']:.3f}  ECE {s['ece']:.3f}")


if __name__ == "__main__":
    main()
