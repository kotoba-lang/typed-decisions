"""Train / evaluate / benchmark backbone A (encoder). `main(argv)` returns the report dict and
writes it to --out/report.json. Everything measured is in the report; nothing is asserted."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch

from .schema import read_jsonl
from .encoder import DecisionEncoder, Collator, load_tokenizer, decision_loss, predict
from .augment import augment, pair
import torch.nn.functional as F
from .metrics import summarize, fit_temperature
from . import bench

H100_USD_PER_S = 0.001097  # modal.com/pricing, read 2026-09-18 — the report multiplies wall time by this, nothing else


def _device(arg: str):
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="answerdotai/ModernBERT-large")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/encoder")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3, help="the scoring head starts from scratch; at the backbone lr it barely moves and its small weights starve the backbone of gradient (measured: 2 epochs at 3e-5 everywhere -> banking77 intent 0.32)")
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--brier-weight", type=float, default=1.0)
    ap.add_argument("--max-state", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N train states (smoke)")
    ap.add_argument("--test-limit", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--bench-n", default="1,10,100")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="random tiny ModernBERT config instead of pretrained (tests)")
    ap.add_argument("--pool", default="span", choices=["opt", "q-opt", "span"])
    ap.add_argument("--augment", type=float, default=0.0, help="probability that a train question is augmented (shuffle / paraphrase / drop / relabel / negate), gold preserved")
    ap.add_argument("--consistency", type=float, default=0.0, help="weight of the symmetric KL between two surface forms of the same question (needs a second forward)")
    ap.add_argument("--no-amp", action="store_true", help="fp32 forward on cuda (isolates bf16 autocast)")
    ap.add_argument("--attn", default="sdpa", help="attn_implementation for the backbone: sdpa | eager")
    ap.add_argument("--reference-compile", default="auto", choices=["auto", "true", "false"], help="ModernBERT config.reference_compile")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = _device(a.device)
    os.makedirs(a.out, exist_ok=True)
    rep: dict = {"backbone": "encoder", "model": a.model, "args": vars(a), "device": str(dev)}

    tok = load_tokenizer(a.model)
    if a.tiny:
        from transformers import ModernBertConfig, ModernBertModel
        cfg = ModernBertConfig(vocab_size=len(tok) + 8, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=1024, pad_token_id=tok.pad_token_id)
        model = DecisionEncoder(ModernBertModel(cfg), 64, a.pool)
    else:
        kw = {} if a.reference_compile == "auto" else {"reference_compile": a.reference_compile == "true"}
        model = DecisionEncoder.from_pretrained(a.model, tok, pool=a.pool, attn_implementation=a.attn, **kw)
    rep["reference_compile"] = getattr(model.backbone.config, "reference_compile", None)
    model.to(dev)
    rep["params_total"] = sum(p.numel() for p in model.parameters())

    train = read_jsonl(os.path.join(a.data, "train.jsonl"))
    val = read_jsonl(os.path.join(a.data, "val.jsonl"))
    test = read_jsonl(os.path.join(a.data, "test.jsonl"))
    if a.limit:
        train = train[: a.limit]
    if a.test_limit:
        test = test[: a.test_limit]
        val = val[: max(50, a.test_limit // 4)]
    rep["data"] = {"train_states": len(train), "train_questions": sum(len(e.questions) for e in train),
                   "test_states": len(test), "test_questions": sum(len(e.questions) for e in test), "val_states": len(val)}

    coll = Collator(tok, max_state_tokens=a.max_state, max_len=a.max_len)
    steps_per_epoch = math.ceil(len(train) / a.batch)
    total_steps = max(1, int(steps_per_epoch * a.epochs))
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": a.lr},
                             {"params": head_params, "lr": a.head_lr}], weight_decay=0.01)
    warm = max(1, int(total_steps * a.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total_steps - s) / max(1, total_steps - warm)) if s >= warm else (s + 1) / warm)
    use_amp = dev.type == "cuda" and not a.no_amp
    rep["autocast_bf16"] = use_amp
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    step, tokens, losses = 0, 0, []
    t0 = time.time()
    order = list(range(len(train)))
    while step < total_steps:
        random.shuffle(order)
        for i in range(0, len(order), a.batch):
            if step >= total_steps:
                break
            chunk = [train[j] for j in order[i : i + a.batch]]
            items = [(e.state, [augment(q, random) if a.augment and random.random() < a.augment else q for q in e.questions]) for e in chunk]
            batch = coll(items, dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(batch["input_ids"], batch["attention_mask"], batch["opt_pos"], batch["opt_mask"], batch["q_pos"], batch["seg"])
            loss, info = decision_loss(logits.float(), batch["gold"], a.brier_weight)
            if a.consistency > 0:
                # second surface form of every question (same gold); symmetric KL on the option distributions,
                # aligned by mapping both back to the ORIGINAL option order (shuffle/drop permute options)
                items2 = [(st, [pair(q, random)[1] for q in qs]) for st, qs in items]
                batch2 = coll(items2, dev)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits2 = model(batch2["input_ids"], batch2["attention_mask"], batch2["opt_pos"], batch2["opt_mask"], batch2["q_pos"], batch2["seg"]).float()
                kl, npairs = 0.0, 0
                for b, ((st, qs), (_, qs2)) in enumerate(zip(items, items2)):
                    for qi, (q, q2) in enumerate(zip(qs, qs2)):
                        if q.kind == "score":  # relabel keeps order; other kinds: align by option text
                            common = list(range(len(q.options)))
                            idx1, idx2 = common, common
                        else:
                            common = [o for o in q.options if o in q2.options]
                            if len(common) < 2:
                                continue
                            idx1 = [q.options.index(o) for o in common]
                            idx2 = [q2.options.index(o) for o in common]
                        p1 = logits[b, qi, idx1].float().log_softmax(-1)
                        p2 = logits2[b, qi, idx2].log_softmax(-1)
                        kl = kl + 0.5 * (F.kl_div(p2, p1, log_target=True, reduction="sum") + F.kl_div(p1, p2, log_target=True, reduction="sum"))
                        npairs += 1
                if npairs:
                    loss = loss + a.consistency * kl / npairs
                    info["kl"] = float(kl.detach() / npairs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tokens += int(batch["attention_mask"].sum())
            losses.append((step, float(loss), info["ce"], info["brier"]))
            if step % 50 == 0:
                print(f"step {step}/{total_steps} loss {loss.item():.4f} ce {info['ce']:.4f} brier {info['brier']:.4f} kl {info.get('kl', 0.0):.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
            step += 1
    train_s = time.time() - t0
    rep["train"] = {"steps": total_steps, "wall_s": train_s, "seq_tokens": tokens, "seq_tokens_per_s": tokens / max(train_s, 1e-9),
                    "loss_first": losses[0][1] if losses else None, "loss_last10_mean": sum(l[1] for l in losses[-10:]) / max(1, len(losses[-10:])),
                    "peak_mem_gib": (torch.cuda.max_memory_allocated() / 2**30) if dev.type == "cuda" else None,
                    "usd_h100": train_s * H100_USD_PER_S if dev.type == "cuda" else None,
                    "train_questions_seen": int(a.epochs * rep["data"]["train_questions"]),
                    "curve": losses[:: max(1, len(losses) // 200)]}
    if dev.type == "cuda":
        rep["train"]["usd_per_1k_train_questions"] = rep["train"]["usd_h100"] / max(1, rep["train"]["train_questions_seen"]) * 1000

    # train-subset metrics: under-fitting vs over-fitting is only readable with both sides
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        rep["metrics_train_subset"] = summarize(predict(model, coll, train[:200], a.eval_batch, dev, temperature=1.0))

    # calibration: temperature on val logits
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        vrec = predict(model, coll, val, a.eval_batch, dev, temperature=1.0)
    T = fit_temperature([r["logits"] for r in vrec], [r["gold"] for r in vrec])
    model.temperature = T
    rep["temperature"] = T

    t1 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        trec = predict(model, coll, test, a.eval_batch, dev, temperature=1.0)
    rep["eval_wall_s"] = time.time() - t1
    rep["metrics_T1"] = summarize(trec)
    import numpy as np
    for r in trec:
        z = np.asarray(r["logits"]) / T
        z = np.exp(z - z.max())
        r["probs"] = (z / z.sum()).tolist()
    rep["metrics_Tfit"] = summarize(trec)

    # OOD questions (never-seen instructions and option sets on the same test states): reads the question or memorised the slot?
    ood_path = os.path.join(a.data, "ood-test.jsonl")
    if os.path.exists(ood_path):
        ood = read_jsonl(ood_path)
        if a.test_limit:
            ood = ood[: a.test_limit]
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                orec = predict(model, coll, ood, a.eval_batch, dev, temperature=T)
            rep["metrics_ood_Tfit"] = summarize(orec)
        except ValueError as e:
            rep["metrics_ood_Tfit"] = {"error": str(e)}
    else:
        rep["metrics_ood_Tfit"] = {"error": "ood-test.jsonl absent"}

    # latency / throughput
    def fn(items):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            b = coll(items, dev)
            return model(b["input_ids"], b["attention_mask"], b["opt_pos"], b["opt_mask"], b["q_pos"], b["seg"])
    def fwd(b):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            return model(b["input_ids"], b["attention_mask"], b["opt_pos"], b["opt_mask"], b["q_pos"], b["seg"])
    prep = lambda items: coll(items, dev)
    ns = [int(n) for n in a.bench_n.split(",") if n]

    def safe(**kw):
        try:
            return bench.latency(**kw)
        except ValueError as e:  # a pack that does not fit this backbone's context is a recorded refusal, not a crash
            return {"n_questions": kw["n_questions"], "max_options": kw.get("max_options"), "timed": "forward" if kw.get("prep") else "e2e", "error": str(e)}
    rep["latency"] = [safe(fn=fn, pool=test, n_questions=n, device=dev, max_options=10) for n in ns]
    rep["latency"] += [safe(fn=fwd, pool=test, n_questions=n, device=dev, max_options=10, prep=prep) for n in ns]
    rep["latency"].append(safe(fn=fn, pool=test, n_questions=10, device=dev, max_options=None))  # incl. the 77-option question
    rep["throughput"] = [bench.throughput(fn, test, b, dev) for b in (1, 8, 32)]

    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    if a.save:
        torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("train",)}, indent=1)[:4000])
    return rep


if __name__ == "__main__":
    main()
