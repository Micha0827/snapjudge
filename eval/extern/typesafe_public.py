"""Answer TypeSafe's public eval cases (evals.typesafe.ai) with this engine.

The 20 public cases (4 workflows, 373 reference decisions) and the published Jev / Opus / Sol
answers are extracted with the scripts of jev-on-a-laptop (github.com/rorshopping/jev-on-a-laptop),
which also score agreement with TypeSafe's reference (consensus of two frontier models).
TypeSafe's raw case data is not redistributed here.

    python eval/extern/typesafe_public.py <full_eval.json> <out.json> --local <model>
    python eval/extern/typesafe_public.py <full_eval.json> <out.json> --url http://127.0.0.1:8724

Output format matches jev-on-a-laptop's results/full-*.json: answers[workflow][case][qid] =
{"raw": value, "kind": type}. Score answers report the most likely level (argmax), since the
reference is an argmax over levels too.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def raw_answer(ans: dict):
    if ans["type"] == "noul":
        return ans["noul"]
    if ans["type"] == "choice":
        return ans["choice"]
    probs = ans["probabilities"]
    return max(probs, key=probs.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full_eval")
    ap.add_argument("out")
    ap.add_argument("--local", help="model path or Hugging Face id (in-process)")
    ap.add_argument("--url", help="base URL of a running server")
    ap.add_argument("--layout", default="auto", choices=["auto", "state_first", "question_first"])
    ap.add_argument("--workflows", nargs="*", help="only these workflows (default: all)")
    args = ap.parse_args()

    if args.local:
        from snapjudge.engine import Engine

        engine = Engine(args.local)
        name = engine.name
        call = lambda state, qs: engine.system_one(state, qs, layout=args.layout)
    else:
        import httpx

        client = httpx.Client(base_url=args.url, timeout=900)
        name = client.get("/v1/models").json()["models"][0]["name"]
        call = lambda state, qs: client.post("/v1/systemone", json={"state": state, "questions": qs, "layout": args.layout}).json()

    full = json.loads(Path(args.full_eval).read_text())
    answers, timings = {}, {}
    for wf, wdata in full["workflows"].items():
        if args.workflows and wf not in args.workflows:
            continue
        answers[wf] = {}
        for case in wdata["cases"]:
            qs = {q["qid"]: {"type": q["type"], "instructions": q["instructions"], "criteria": q["criteria"]}
                  for q in case["questions"]}
            t = time.perf_counter()
            res = call(case["input_text"], qs)
            ms = (time.perf_counter() - t) * 1000
            answers[wf][case["case_id"]] = {qid: {"raw": raw_answer(a), "kind": a["type"]}
                                            for qid, a in res["answers"].items()}
            timings[f"{wf}/{case['case_id']}"] = {"total_ms": round(ms, 1), "questions": len(qs),
                                                  "input_tokens": res["usage"]["input_tokens"]}
            print(f"{wf:28s} {case['case_id'][:40]:40s} {len(qs):3d} questions "
                  f"{res['usage']['input_tokens']:6d} tokens {ms / 1000:6.1f} s", flush=True)
    Path(args.out).write_text(json.dumps({"model": name, "timings": timings, "answers": answers}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
