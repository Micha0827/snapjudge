"""Run a labeled test set against a model: in-process, or over HTTP against any System One endpoint.

In-process:   python eval/run_eval.py --local mlx-community/Qwen3.6-35B-A3B-4bit
Local server: python eval/run_eval.py --url http://127.0.0.1:8724 --tag server
TypeSafe:     python eval/run_eval.py --url https://api.typesafe.ai --model jev-latest \\
                  --key-env TYPESAFE_API_KEY --tag jev

Raw answers go to results/<tag>.jsonl (input for calibrate.py), metrics per question to
results/<tag>.summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from metrics import decisions, errors, summarize  # noqa: E402


def make_caller(args):
    if args.local:
        from snapjudge.engine import Engine

        engine = Engine(os.path.expanduser(args.local), calibration_path=args.calibration, adapter_path=args.adapter)
        return engine.name, lambda state, qs: engine.system_one(state, qs, layout=args.layout)

    import httpx

    headers = {"Content-Type": "application/json"}
    if args.key_env:
        headers["Authorization"] = f"Bearer {os.environ[args.key_env]}"
    client = httpx.Client(base_url=args.url, headers=headers, timeout=120)

    def call(state, qs):
        for attempt in range(5):
            body = {"state": state, "model": args.model, "questions": qs}
            if args.layout != "auto":
                body["layout"] = args.layout  # only the local server knows this field
            r = client.post("/v1/systemone", json=body)
            if r.status_code in (429, 529):
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()

    return args.model, call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", help="model path or Hugging Face id; runs the engine in-process")
    ap.add_argument("--url", help="base URL of a System One endpoint")
    ap.add_argument("--model", default="local")
    ap.add_argument("--key-env", help="environment variable that holds the API key")
    ap.add_argument("--calibration", help="temperature file for the local engine")
    ap.add_argument("--adapter", help="LoRA adapter directory for the local engine")
    ap.add_argument("--testset", default=str(ROOT / "eval" / "testset_de.json"))
    ap.add_argument("--tag")
    ap.add_argument("--layout", default="auto", choices=["auto", "state_first", "question_first", "header"])
    ap.add_argument("--direction", action="store_true",
                    help="add 'richtung' (incoming/outgoing) to bank transactions, computed from the sign in code")
    args = ap.parse_args()
    if not (args.local or args.url):
        ap.error("pass --local or --url")

    testset = json.loads(Path(args.testset).read_text())
    if args.direction:
        for item in testset["items"]:
            if isinstance(item["state"], dict) and "betrag" in item["state"]:
                st = item["state"]
                item["state"] = {"richtung": "Eingang" if st["betrag"] > 0 else "Ausgang",
                                 "betrag_eur": abs(st["betrag"]), **{k: v for k, v in st.items() if k != "betrag"}}
    name, call = make_caller(args)
    tag = args.tag or name
    print(f"Model: {name}", flush=True)

    first = testset["items"][0]
    call(first["state"], testset["domains"][first["domain"]])  # warm-up, not measured

    records, latencies = [], []
    for item in testset["items"]:
        qs = testset["domains"][item["domain"]]
        t = time.perf_counter()
        res = call(item["state"], qs)
        ms = (time.perf_counter() - t) * 1000
        latencies.append(ms)
        records.append({"id": item["id"], "domain": item["domain"], "gold": item["gold"],
                        "answers": res["answers"], "usage": res.get("usage"), "latency_ms": round(ms, 1)})

    (ROOT / "results").mkdir(exist_ok=True)
    out = ROOT / "results" / f"{tag}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))

    decs = decisions(records, testset)
    summary = {"model": name, "latency_ms": {
        "median": round(statistics.median(latencies), 1),
        "p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1),
    }, "overall": summarize(decs), "per_question": {}}
    for qid in sorted({d["qid"] for d in decs}):
        summary["per_question"][qid] = summarize([d for d in decs if d["qid"] == qid])
    summary["errors"] = errors(decs)
    (ROOT / "results" / f"{tag}.summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    g = summary["overall"]
    print(f"Latency per request: median {summary['latency_ms']['median']} ms, p95 {summary['latency_ms']['p95']} ms")
    print(f"Overall n={g['n']}: accuracy {g['acc']:.1%} | mean confidence {g['mean_conf']:.1%} | "
          f"ECE {g['ece']:.3f} | Brier {g['brier']:.3f} | NLL {g['nll']:.3f} | "
          f"p≥0.9: {g['auto_share']:.0%} of cases, {g['auto_acc']:.1%} of those correct")
    print(f"{'Question':22s} {'n':>3s} {'acc':>7s} {'conf':>6s} {'ECE':>6s} {'NLL':>6s}")
    for qid, s in summary["per_question"].items():
        print(f"{qid:22s} {s['n']:3d} {s['acc']:7.1%} {s['mean_conf']:6.1%} {s['ece']:6.3f} {s['nll']:6.3f}")
    if summary["errors"]:
        print("Errors:")
        for e in summary["errors"]:
            print(f"  {e['item']:10s} {e['qid']:20s} expected {e['gold']!s:14s} got {e['got']!s:14s} "
                  f"p={e['p_got']} (expected p={e['p_gold']})")


if __name__ == "__main__":
    main()
