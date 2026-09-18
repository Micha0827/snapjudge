"""Print all results/*.summary.json as one Markdown table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
    print("| Run | Accuracy | Mean confidence | ECE | Brier | NLL | Share p ≥ 0.9 | of those correct | Median latency |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for f in sorted(folder.glob("*.summary.json")):
        s = json.loads(f.read_text())
        g = s.get("overall") or s["gesamt"]  # older lab runs used German keys
        print(f"| {f.name.removesuffix('.summary.json')} | {g['acc']:.1%} | {g['mean_conf']:.1%} | {g['ece']:.3f} | "
              f"{g['brier']:.3f} | {g['nll']:.3f} | {g['auto_share']:.0%} | {g['auto_acc']:.1%} | "
              f"{s['latency_ms']['median'] / 1000:.2f} s |")


if __name__ == "__main__":
    main()
