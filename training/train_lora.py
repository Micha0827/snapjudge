"""Fine-tune a Qwen 3.5-family model with LoRA on MLX for typed decisions.

The model is trained on exactly what the engine reads at inference time: the probability of
each allowed answer label at the first answer position, with the prompt built by the engine
itself (question first). The target is the full gold distribution, not only its argmax, so the
loss is a soft cross-entropy (a proper scoring rule) that rewards honest probabilities.

    python training/train_lora.py --model mlx-community/Qwen3.5-2B-MLX-4bit \\
        --train train.parquet --out adapters/qwen3.5-2b-td

Training data: rows in the LocalLLaMA/typed-decisions format (columns state, questions, gold,
id, workflow), e.g. that dataset's `train` split (Apache-2.0).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snapjudge.engine import Engine  # noqa: E402
from snapjudge.prompts import question_block  # noqa: E402


def gold_distribution(qtype: str, options: list[str], gold: dict) -> list[float]:
    probs = gold["probabilities"]
    if qtype == "noul":  # the engine's noul labels are yes / no
        probs = {"yes": probs.get("true", 0.0), "no": probs.get("false", 0.0)}
    dist = [float(probs.get(o, 0.0)) for o in options]
    total = sum(dist)
    return [d / total for d in dist] if total > 0 else [1 / len(dist)] * len(dist)


def build_examples(engine: Engine, rows, max_len: int):
    examples, skipped = [], {"too_long": 0, "shared_first_token": 0}
    for r in rows:
        state, questions, gold = json.loads(r["state"]), json.loads(r["questions"]), json.loads(r["gold"])
        for qid, q in questions.items():
            text, options, surfaces = question_block(q)
            head, tail = engine._render(state, text, "question_first")
            ids = engine._encode(head + tail)
            if len(ids) > max_len:
                skipped["too_long"] += 1
                continue
            tail_ids = engine._encode(tail)
            groups = [sorted({engine._label_tokens(tail, tail_ids, f)[0] for f in surfaces[o]}) for o in options]
            firsts = [t for g in groups for t in g]
            if len(firsts) != len(set(firsts)):  # labels not separable by their first token
                skipped["shared_first_token"] += 1
                continue
            examples.append({"ids": ids, "groups": groups, "target": gold_distribution(q["type"], options, gold[qid]),
                             "case": r["id"], "workflow": r["workflow"], "type": q["type"]})
    return examples, skipped


def batches(examples, batch_size: int, shuffle: bool, rng: random.Random):
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["ids"]))
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(chunks)
    for chunk in chunks:
        yield [examples[i] for i in chunk]


def collate(batch, pad_id: int):
    width = max(len(e["ids"]) for e in batch)
    tokens = mx.array([e["ids"] + [pad_id] * (width - len(e["ids"])) for e in batch], dtype=mx.int32)
    last = mx.array([len(e["ids"]) - 1 for e in batch])
    n_opt = max(len(e["groups"]) for e in batch)
    n_form = max(len(g) for e in batch for g in e["groups"])
    tok = [[[0] * n_form for _ in range(n_opt)] for _ in batch]
    form_mask = [[[False] * n_form for _ in range(n_opt)] for _ in batch]
    opt_mask = [[False] * n_opt for _ in batch]
    target = [[0.0] * n_opt for _ in batch]
    for b, e in enumerate(batch):
        for o, g in enumerate(e["groups"]):
            opt_mask[b][o] = True
            target[b][o] = e["target"][o]
            for f, t in enumerate(g):
                tok[b][o][f] = t
                form_mask[b][o][f] = True
    return tokens, last, mx.array(tok), mx.array(form_mask), mx.array(opt_mask), mx.array(target)


def option_log_probs(engine: Engine, tokens, last, tok, form_mask, opt_mask):
    lm = engine.lm
    B, L = tokens.shape
    pos = mx.broadcast_to(mx.arange(L, dtype=mx.int32)[None, None, :], (3, B, L))
    out = lm(tokens, cache=None, position_ids=pos, return_hidden=True, skip_logits=True)
    h = out.hidden_states[-1][mx.arange(B), last][:, None, :].astype(mx.float32)
    logits = (lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h))[:, 0, :]
    picked = mx.take_along_axis(logits, tok.reshape(B, -1), axis=1).reshape(tok.shape)
    picked = mx.where(form_mask, picked, -1e9)
    opt_logits = mx.logsumexp(picked, axis=-1)  # surface forms of one option add up
    opt_logits = mx.where(opt_mask, opt_logits, -1e9)
    return opt_logits - mx.logsumexp(opt_logits, axis=-1, keepdims=True)


def soft_cross_entropy(logp, target, opt_mask):
    return -(mx.where(opt_mask, target * logp, 0.0)).sum(axis=-1).mean()


def evaluate(engine, examples, batch_size, pad_id):
    engine.model.eval()
    total, n, correct = 0.0, 0, 0
    for batch in batches(examples, batch_size, False, random.Random(0)):
        tokens, last, tok, form_mask, opt_mask, target = collate(batch, pad_id)
        logp = option_log_probs(engine, tokens, last, tok, form_mask, opt_mask)
        loss = soft_cross_entropy(logp, target, opt_mask)
        mx.eval(loss, logp)
        total += loss.item() * len(batch)
        n += len(batch)
        pred = mx.argmax(logp, axis=-1).tolist()
        gold = mx.argmax(target, axis=-1).tolist()
        correct += sum(p == g for p, g in zip(pred, gold))
    engine.model.train()
    return total / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True, help="parquet in typed-decisions format")
    ap.add_argument("--out", required=True, help="adapter output directory")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-cases", type=int, help="use only the first N cases (smoke test)")
    args = ap.parse_args()

    import pandas as pd
    from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model, grad_checkpoint, save_adapter

    rng = random.Random(args.seed)
    mx.random.seed(args.seed)
    engine = Engine(args.model)
    pad_id = engine.end_id

    df = pd.read_parquet(args.train)
    if args.limit_cases:
        df = df.head(args.limit_cases)
    cases = sorted(df["id"].unique())
    rng.shuffle(cases)
    val_cases = set(cases[: int(len(cases) * args.val_frac)])
    train_rows = [r for _, r in df.iterrows() if r["id"] not in val_cases]
    val_rows = [r for _, r in df.iterrows() if r["id"] in val_cases]
    train_ex, skipped = build_examples(engine, train_rows, args.max_len)
    val_ex, _ = build_examples(engine, val_rows, args.max_len)
    print(f"examples: train {len(train_ex)}, val {len(val_ex)}, skipped {skipped}", flush=True)

    targets = [n for n in find_all_linear_names(engine.model.language_model) if n != "lm_head"]
    get_peft_model(engine.model, targets, rank=args.rank, alpha=args.alpha, dropout=0.0)
    if args.grad_checkpoint:
        grad_checkpoint(engine.lm.model.layers[0])
    engine.model.train()  # also switches the linear-attention layers to their differentiable path

    steps_per_epoch = math.ceil(len(train_ex) / args.batch)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup = max(1, total_steps // 20)
    schedule = optim.join_schedules(
        [optim.linear_schedule(args.lr / 10, args.lr, warmup), optim.cosine_decay(args.lr, total_steps - warmup, args.lr / 20)],
        [warmup],
    )
    opt = optim.AdamW(learning_rate=schedule, weight_decay=0.0)

    def loss_fn(model, tokens, last, tok, form_mask, opt_mask, target):
        return soft_cross_entropy(option_log_probs(engine, tokens, last, tok, form_mask, opt_mask), target, opt_mask)

    loss_and_grad = nn.value_and_grad(engine.model, loss_fn)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_loss, val_acc = evaluate(engine, val_ex, 8, pad_id)
    print(f"step 0: val loss {val_loss:.4f} acc {val_acc:.3f}", flush=True)
    best = val_loss
    log = [{"step": 0, "val_loss": val_loss, "val_acc": val_acc}]

    step, t0, running = 0, time.perf_counter(), []
    while step < total_steps:
        for batch in batches(train_ex, args.batch, True, rng):
            tokens, last, tok, form_mask, opt_mask, target = collate(batch, pad_id)
            loss, grads = loss_and_grad(engine.model, tokens, last, tok, form_mask, opt_mask, target)
            opt.update(engine.model, grads)
            mx.eval(engine.model.parameters(), opt.state, loss)
            running.append(loss.item())
            step += 1
            if step % 25 == 0:
                rate = (time.perf_counter() - t0) / step
                print(f"step {step}/{total_steps}: train loss {sum(running[-25:]) / len(running[-25:]):.4f} "
                      f"| {rate:.1f} s/step | eta {rate * (total_steps - step) / 60:.0f} min | "
                      f"peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
            if step % args.eval_every == 0 or step == total_steps:
                val_loss, val_acc = evaluate(engine, val_ex, 8, pad_id)
                log.append({"step": step, "val_loss": val_loss, "val_acc": val_acc})
                mark = ""
                if val_loss < best:
                    best = val_loss
                    save_adapter(engine.model, out_dir / "adapters.safetensors")
                    mark = " (saved)"
                print(f"step {step}: val loss {val_loss:.4f} acc {val_acc:.3f}{mark}", flush=True)
            if step >= total_steps:
                break
    if not (out_dir / "adapters.safetensors").exists():
        save_adapter(engine.model, out_dir / "adapters.safetensors")
        print("note: validation loss never improved; saved the final adapter anyway", flush=True)
    (out_dir / "training_log.json").write_text(json.dumps({"args": vars(args), "skipped": skipped, "log": log}, indent=2))
    n_train = sum(v.size for _, v in tree_flatten(engine.model.trainable_parameters()))
    print(f"done: best val loss {best:.4f}, trainable params {n_train / 1e6:.1f} M, adapter in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
