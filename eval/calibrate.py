"""Fit a temperature per question type on saved results (temperature scaling).

    python eval/calibrate.py results/qwen3.6-35b-a3b.jsonl --name Qwen3.6-35B-A3B-4bit [--write]

Reports 2-fold cross-validation (fit on one half of the items, measure on the other). With
--write, the temperature fitted on all items goes to calibration.json ({model_name: {type: T}}),
which the engine reads via SO_CALIBRATION. On the small synthetic test set the fitted
temperature sometimes worsened cross-validated ECE (too few errors for a stable fit), so
nothing is applied automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from metrics import decisions, summarize  # noqa: E402

GRID = [round(0.5 + 0.1 * i, 2) for i in range(76)]  # 0.5 … 8.0


def fit(decs: list[dict]) -> float:
    return min(GRID, key=lambda t: summarize(decs, t)["nll"]) if decs else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--name", help="model name as reported by the engine (default: from summary.json)")
    ap.add_argument("--testset", default=str(ROOT / "eval" / "testset_de.json"))
    ap.add_argument("--out", default=str(ROOT / "calibration.json"))
    ap.add_argument("--write", action="store_true", help="save the temperatures to --out")
    args = ap.parse_args()

    testset = json.loads(Path(args.testset).read_text())
    records = [json.loads(line) for line in Path(args.results).read_text().splitlines()]
    decs = decisions(records, testset)
    name = args.name or json.loads(Path(args.results).with_suffix(".summary.json").read_text())["model"]

    items = sorted({d["item"] for d in decs})
    folds = [set(items[0::2]), set(items[1::2])]
    types = sorted({d["type"] for d in decs})

    temps, cv_rows = {}, []
    for t in types:
        sub = [d for d in decs if d["type"] == t]
        temps[t] = fit(sub)
        held = []
        for k in range(2):
            train = [d for d in sub if d["item"] not in folds[k]]
            test = [d for d in sub if d["item"] in folds[k]]
            tk = fit(train)
            held += [dict(d, _t=tk) for d in test]
        before = summarize(sub, 1.0)
        # Cross-validated: each decision uses the temperature fitted on the other half
        after_cv = summarize(held)
        cv_rows.append((t, len(sub), temps[t], before, after_cv))

    print(f"Model: {name}")
    print(f"{'Type':7s} {'n':>4s} {'T':>5s} | {'ECE raw':>10s} {'ECE CV':>7s} | {'NLL raw':>10s} {'NLL CV':>7s} | "
          f"{'conf raw':>12s} {'conf CV':>8s} | accuracy")
    for t, n, temp, b, a in cv_rows:
        print(f"{t:7s} {n:4d} {temp:5.2f} | {b['ece']:10.3f} {a['ece']:7.3f} | {b['nll']:10.3f} {a['nll']:7.3f} | "
              f"{b['mean_conf']:12.1%} {a['mean_conf']:8.1%} | {b['acc']:.1%}")

    if not args.write:
        print(f"Fitted (not saved; use --write): {temps}")
        return
    out = Path(args.out)
    data = json.loads(out.read_text()) if out.exists() else {}
    data[name] = temps
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved temperatures to {out}: {temps}")


if __name__ == "__main__":
    main()
