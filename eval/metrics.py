"""Metrics for decisions with probabilities: accuracy, ECE, Brier score, NLL."""

from __future__ import annotations

import math


def decisions(records: list[dict], testset: dict) -> list[dict]:
    """Flatten result records into a list of scored decisions."""
    out = []
    for rec in records:
        questions = testset["domains"][rec["domain"]]
        for qid, gold in rec["gold"].items():
            ans = rec["answers"][qid]
            qtype = questions[qid]["type"]
            if qtype == "noul":
                probs, gold_key = {"yes": ans["noul"], "no": 1 - ans["noul"]}, "yes" if gold else "no"
            elif qtype == "score":
                probs, gold_key = ans["probabilities"], str(gold)
            else:
                probs, gold_key = ans["probabilities"], gold
            out.append({"item": rec["id"], "qid": f"{rec['domain']}.{qid}", "type": qtype,
                        "probs": probs, "gold": gold_key, "score": ans.get("score")})
    return out


def temper(probs: dict, temp: float) -> dict:
    """Apply a temperature to a distribution: p^(1/T), renormalized."""
    if temp == 1.0:
        return probs
    logs = {k: math.log(max(v, 1e-12)) / temp for k, v in probs.items()}
    m = max(logs.values())
    ex = {k: math.exp(v - m) for k, v in logs.items()}
    s = sum(ex.values())
    return {k: v / s for k, v in ex.items()}


def summarize(decs: list[dict], temp: float | dict = 1.0, bins: int = 10, gate: float = 0.9) -> dict:
    if not decs:
        return {}
    n = len(decs)
    correct, confs, brier, nll, mae = [], [], 0.0, 0.0, []
    for d in decs:
        # Precedence: temperature on the decision itself (cross-validation), then per type, then global
        t = d.get("_t", temp.get(d["type"], 1.0) if isinstance(temp, dict) else temp)
        p = temper(d["probs"], t)
        top = max(p, key=p.get)
        correct.append(top == d["gold"])
        confs.append(p[top])
        brier += sum((v - (k == d["gold"])) ** 2 for k, v in p.items())
        nll += -math.log(max(p.get(d["gold"], 0.0), 1e-6))
        if d["type"] == "score":
            mae.append(abs(sum(int(k) * v for k, v in p.items()) - int(d["gold"])))
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            acc_b = sum(correct[i] for i in idx) / len(idx)
            conf_b = sum(confs[i] for i in idx) / len(idx)
            ece += len(idx) / n * abs(acc_b - conf_b)
    gated = [i for i, c in enumerate(confs) if c >= gate]
    res = {
        "n": n,
        "acc": sum(correct) / n,
        "mean_conf": sum(confs) / n,
        "ece": ece,
        "brier": brier / n,
        "nll": nll / n,
        "auto_share": len(gated) / n,
        "auto_acc": (sum(correct[i] for i in gated) / len(gated)) if gated else float("nan"),
    }
    if mae:
        res["score_mae"] = sum(mae) / len(mae)
    return res


def errors(decs: list[dict]) -> list[dict]:
    out = []
    for d in decs:
        top = max(d["probs"], key=d["probs"].get)
        if top != d["gold"]:
            out.append({"item": d["item"], "qid": d["qid"], "gold": d["gold"], "got": top,
                        "p_got": round(d["probs"][top], 3), "p_gold": round(d["probs"].get(d["gold"], 0), 3)})
    return out
