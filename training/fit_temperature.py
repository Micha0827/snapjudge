"""Fit one temperature per question type on the held-out validation cases of a training run.

Reproduces the validation split of train_lora.py (same seed and fraction), answers those cases
with the trained model, and picks per type the temperature that minimizes the log loss of the
gold label. Writes {model_name: {type: T}} into a calibration file the engine reads.

    python training/fit_temperature.py --model <base> --adapter adapters/<run> --train train.parquet \\
        --out calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snapjudge.engine import Engine  # noqa: E402

GRID = [round(0.1 + 0.05 * i, 2) for i in range(59)]  # 0.10 … 3.00


def temper(probs: dict, t: float) -> dict:
    logs = {k: math.log(max(v, 1e-9)) / t for k, v in probs.items()}
    m = max(logs.values())
    ex = {k: math.exp(v - m) for k, v in logs.items()}
    z = sum(ex.values())
    return {k: v / z for k, v in ex.items()}


def nll(rows, t):
    return sum(-math.log(max(temper(p, t).get(g, 0.0), 1e-9)) for p, g in rows) / len(rows)


def ece(rows, t, bins=10):
    pts = []
    for p, g in rows:
        q = temper(p, t)
        top = max(q, key=q.get)
        pts.append((q[top], top == g))
    total = 0.0
    for b in range(bins):
        sel = [x for x in pts if b / bins < x[0] <= (b + 1) / bins]
        if sel:
            total += len(sel) / len(pts) * abs(sum(c for _, c in sel) / len(sel) - sum(c for c, _ in sel) / len(sel))
    return total


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(args.train)
    cases = sorted(df["id"].unique())
    random.Random(args.seed).shuffle(cases)
    val_cases = set(cases[: int(len(cases) * args.val_frac)])
    val = df[df["id"].isin(val_cases)]

    engine = Engine(args.model, adapter_path=args.adapter)
    rows = {}
    for _, r in val.iterrows():
        questions, gold = json.loads(r["questions"]), json.loads(r["gold"])
        res = engine.system_one(json.loads(r["state"]), questions)
        for qid, q in questions.items():
            a = res["answers"][qid]
            probs = {"true": a["noul"], "false": 1 - a["noul"]} if a["type"] == "noul" else a["probabilities"]
            rows.setdefault(q["type"], []).append((probs, str(gold[qid]["label"])))

    temps = {}
    for qtype, data in sorted(rows.items()):
        t = min(GRID, key=lambda x: nll(data, x))
        temps[qtype] = t
        print(f"{qtype:7s} n={len(data):4d}  T={t:.2f}  NLL {nll(data, 1.0):.3f} -> {nll(data, t):.3f}  "
              f"ECE {ece(data, 1.0):.3f} -> {ece(data, t):.3f}", flush=True)

    out = Path(args.out)
    data = json.loads(out.read_text()) if out.exists() else {}
    data[engine.name] = temps
    out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"saved for {engine.name}: {temps}")


if __name__ == "__main__":
    main()
