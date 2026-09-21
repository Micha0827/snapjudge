"""Game sentences without the game: accuracy and latency against a running server.

    python eval/runner_labels.py [--url http://127.0.0.1:8724] [--lang en|de] [--german-labels] [--layout …]

--german-labels (only with --lang de) swaps run/jump/long_jump/duck for laufen/springen/…
"""

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import httpx

GAME = Path(__file__).resolve().parent.parent / "snapjudge" / "game"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8724")
    ap.add_argument("--lang", default="en", choices=["en", "de"])
    ap.add_argument("--german-labels", action="store_true")
    ap.add_argument("--layout", default="auto", choices=["auto", "state_first", "question_first", "header"])
    args = ap.parse_args()
    data = json.loads((GAME / f"obstacles.{args.lang}.json").read_text())
    question, back = data["question"], {k: k for k in data["question"]["criteria"]}
    if args.german_labels:
        de = data["labels_de"]
        question = {**question, "criteria": {de[k]: v for k, v in question["criteria"].items()}}
        back = {v: k for k, v in de.items()}

    client = httpx.Client(base_url=args.url, timeout=60)
    ms, errors, confusion, per_kind = [], [], Counter(), Counter()
    for h in data["obstacles"]:
        t = time.perf_counter()
        a = client.post("/v1/systemone", json={"state": h["text"], "questions": {"action": question}, "layout": args.layout}).json()["answers"]["action"]
        ms.append((time.perf_counter() - t) * 1000)
        got = back[a["choice"]]
        per_kind[(h["kind"], got == h["label"])] += 1
        if got != h["label"]:
            confusion[(h["label"], got)] += 1
            errors.append(f"    {h['label']:9s} -> {got:9s} p={a['probabilities'][a['choice']]:.2f}  {h['text']}")
    n = len(data["obstacles"])
    right = n - len(errors)
    kinds = " · ".join(f"{k} {per_kind[(k, True)]}/{per_kind[(k, True)] + per_kind[(k, False)]}" for k in ("normal", "tricky", "trap"))
    print(f"{args.lang}{' (German labels)' if args.german_labels else ''}, {args.layout}: {right}/{n} correct ({right / n:.0%}) | {kinds} | "
          f"median latency {statistics.median(ms):.0f} ms")
    if confusion:
        print("  confusions:", dict(confusion))
        print("\n".join(errors))


if __name__ == "__main__":
    main()
