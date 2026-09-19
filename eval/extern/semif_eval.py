"""SemIf authored144 (github.com/TheoLeeCJ/SemIf, MIT) against this engine: mean per-family balanced accuracy, as SemIf reports it.

    python eval/extern/semif_eval.py http [url]      # against a running server (default :8724)
    python eval/extern/semif_eval.py local <model> [adapter]   # in-process
"""
import json, sys, time, statistics
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
rows = [json.loads(l) for l in open(Path(__file__).with_name("semif_authored144.jsonl"))]

def to_q(r):
    return {"type": "choice", "instructions": r["question"],
            "criteria": {o["id"]: o["description"] for o in r["options"]}}

def run(call, name):
    preds, ms = [], []
    for r in rows:
        t = time.perf_counter(); a = call(r["state"], {"q": to_q(r)})["answers"]["q"]; ms.append((time.perf_counter()-t)*1000)
        preds.append((r["family"], r["options"][r["label"]]["id"], a["choice"], a["probabilities"][a["choice"]]))
    fam = defaultdict(list)
    for f, g, p, _ in preds: fam[f].append((g, p))
    bas = {}
    for f, pairs in fam.items():
        classes = sorted({g for g, _ in pairs})
        bas[f] = statistics.mean(sum(1 for g, p in pairs if g == c and p == c) / sum(1 for g, _ in pairs if g == c) for c in classes)
    acc = sum(g == p for _, g, p, _ in preds) / len(preds)
    hi = [(g == p) for _, g, p, pr in preds if pr >= 0.9]
    print(f"{name}: mean family balanced accuracy {statistics.mean(bas.values()):.3f} | accuracy {acc:.3f} | "
          + " · ".join(f"{f} {v:.3f}" for f, v in bas.items())
          + f" | p≥0.9: {len(hi)/len(preds):.0%} of cases, {sum(hi)/max(1,len(hi)):.1%} of those correct | median {statistics.median(ms):.0f} ms", flush=True)
    return preds

mode = sys.argv[1]
if mode == "http":
    import httpx
    c = httpx.Client(base_url=sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8724", timeout=60)
    name = c.get("/v1/models").json()["models"][0]["name"]
    run(lambda s, q: c.post("/v1/systemone", json={"state": s, "questions": q}).json(), name)
else:
    from snapjudge.engine import Engine
    e = Engine(sys.argv[2], adapter_path=sys.argv[3] if len(sys.argv) > 3 else None)
    e.system_one("warm", {"q": to_q(rows[0])})
    run(lambda s, q: e.system_one(s, q), e.name)
