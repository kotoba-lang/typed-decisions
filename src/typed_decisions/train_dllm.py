"""Train (LoRA) / evaluate / benchmark backbone B (LLaDA-family dLLM). Same report shape as
train_encoder so the two are read side by side. Also evaluates the untrained model first
(zero-shot through the same slot read-out), which is the "train cost = 0" row."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch

from .schema import read_jsonl
from .dllm import DllmDecider, OptionAlphabet, predict, find_lora_targets
from .metrics import summarize, fit_temperature
from .train_encoder import H100_USD_PER_S, _device
from . import bench


def load(model_name: str, dev, dtype, tiny: bool):
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tiny:
        from transformers import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(vocab_size=len(tok) + 8, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=2048)
        model = LlamaForCausalLM(cfg)
        # a causal LM stands in for the bidirectional dLLM in tests: masked slots after the prompt
        # still see the prompt, which is all the mechanics need
        if tok.mask_token_id is None:
            tok.add_special_tokens({"mask_token": "[MASK]"})
            model.resize_token_embeddings(len(tok))
    else:
        try:
            from transformers import AutoConfig, AutoModelForMaskedLM
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            if getattr(cfg, "model_type", "") in ("modernbert", "bert", "roberta", "deberta-v2"):
                model = AutoModelForMaskedLM.from_pretrained(model_name, dtype=dtype)  # an encoder MLM is a 1-step "dLLM"
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=dtype)
        except Exception as e:  # LLaDA-8B registers only AutoModel
            print("AutoModelForCausalLM failed, trying AutoModel:", repr(e)[:200])
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True, dtype=dtype)
    model.to(dev)
    return tok, model


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="inclusionAI/LLaDA-MoE-7B-A1B-Instruct")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/dllm")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--brier-weight", type=float, default=1.0)
    ap.add_argument("--max-state", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--no-lora", action="store_true", help="full fine-tune (tiny/tests)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--test-limit", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--bench-n", default="1,10,100")
    ap.add_argument("--eval-steps", default="1,2,4")
    ap.add_argument("--skip-zero-shot", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = _device(a.device)
    dtype = getattr(torch, a.dtype) if dev.type == "cuda" else torch.float32
    os.makedirs(a.out, exist_ok=True)
    rep: dict = {"backbone": "dllm", "model": a.model, "args": vars(a), "device": str(dev)}

    tok, model = load(a.model, dev, dtype, a.tiny)
    alphabet = OptionAlphabet(tok)
    mask_id = DllmDecider.resolve_mask_id(tok, model)
    dec = DllmDecider(model, tok, alphabet, mask_id, max_state_tokens=a.max_state)
    rep["alphabet_size"] = len(alphabet)
    rep["mask_id"] = mask_id
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

    # fp32 weights + bf16 autocast for a full fine-tune (encoder-MLM route); bf16 weights for LoRA on the 7B
    use_amp = dev.type == "cuda" and a.dtype == "float32"
    autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)
    rep["autocast_bf16"] = use_amp

    def evaluate(tag: str, steps: int, temperature: float = 1.0):
        model.eval()
        t1 = time.time()
        with autocast():
            rec = predict(dec, test, a.eval_batch, dev, steps=steps, temperature=temperature)
        rep[f"metrics_{tag}"] = summarize(rec)
        rep[f"metrics_{tag}"]["_eval_wall_s"] = time.time() - t1
        return rec

    if not a.skip_zero_shot:
        evaluate("zero_shot_steps1", 1)

    # ---- LoRA
    if not a.no_lora:
        from peft import LoraConfig, get_peft_model
        targets = find_lora_targets(model)
        if not targets:
            raise RuntimeError("REFUSE: no LoRA target modules found in " + a.model)
        model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, target_modules=targets))
        dec.model = model
        rep["lora_targets"] = targets
    rep["params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    steps_per_epoch = math.ceil(len(train) / (a.batch * a.grad_accum))
    total_steps = max(1, int(steps_per_epoch * a.epochs))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    warm = max(1, int(total_steps * a.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total_steps - s) / max(1, total_steps - warm)))
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    step, tokens, losses = 0, 0, []
    t0 = time.time()
    order = list(range(len(train)))
    while step < total_steps:
        random.shuffle(order)
        for i in range(0, len(order), a.batch * a.grad_accum):
            if step >= total_steps:
                break
            opt.zero_grad(set_to_none=True)
            acc_loss, acc_info = 0.0, {"ce": 0.0, "brier": 0.0, "n": 0}
            for k in range(a.grad_accum):
                chunk = [train[j] for j in order[i + k * a.batch : i + (k + 1) * a.batch]]
                if not chunk:
                    break
                input_ids, attn, slots, opt_ids = dec.collate([(e.state, e.questions) for e in chunk], dev)
                golds = [[q.gold for q in e.questions] for e in chunk]
                with autocast():
                    loss, info = dec.loss(input_ids, attn, slots, opt_ids, golds, a.brier_weight)
                (loss / a.grad_accum).backward()
                tokens += int(attn.sum())
                acc_loss += float(loss) / a.grad_accum
                for kk in ("ce", "brier"):
                    acc_info[kk] += info[kk] / a.grad_accum
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            losses.append((step, acc_loss, acc_info["ce"], acc_info["brier"]))
            if step % 20 == 0:
                print(f"step {step}/{total_steps} loss {acc_loss:.4f} ce {acc_info['ce']:.4f} brier {acc_info['brier']:.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
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

    model.eval()
    with autocast():
        rep["metrics_train_subset"] = summarize(predict(dec, train[:200], a.eval_batch, dev, steps=1))
        vrec = predict(dec, val, a.eval_batch, dev, steps=1)
    T = fit_temperature([r["logits"] for r in vrec], [r["gold"] for r in vrec])
    rep["temperature"] = T
    for s in [int(x) for x in a.eval_steps.split(",") if x]:
        evaluate(f"trained_steps{s}_T1", s)
    evaluate("trained_steps1_Tfit", 1, temperature=T)
    ood_path = os.path.join(a.data, "ood-test.jsonl")
    if os.path.exists(ood_path):
        ood = read_jsonl(ood_path)
        if a.test_limit:
            ood = ood[: a.test_limit]
        with autocast():
            rep["metrics_ood_Tfit"] = summarize(predict(dec, ood, a.eval_batch, dev, steps=1, temperature=T))
    else:
        rep["metrics_ood_Tfit"] = {"error": "ood-test.jsonl absent"}

    def fn_steps(steps):
        def f(items):
            with autocast():
                return dec.decide(items, dev, steps=steps)
        return f
    def fwd(b):
        input_ids, attn, slots, opt_ids = b
        with autocast():
            return dec.slot_logits(input_ids, attn, slots, opt_ids)
    prep = lambda items: dec.collate(items, dev)
    ns = [int(n) for n in a.bench_n.split(",") if n]
    rep["latency"] = [dict(bench.latency(fn_steps(1), test, n, dev, max_options=10), steps=1) for n in ns]
    rep["latency"] += [dict(bench.latency(fwd, test, n, dev, max_options=10, prep=prep), steps=1) for n in ns]
    rep["latency"].append(dict(bench.latency(fn_steps(1), test, 10, dev, max_options=None), steps=1))  # incl. the 77-option question
    rep["latency"] += [dict(bench.latency(fn_steps(s), test, 10, dev, max_options=10), steps=s) for s in (2, 4)]
    rep["throughput"] = [bench.throughput(fn_steps(1), test, b, dev, max_examples=128) for b in (1, 8)]

    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("train",)}, indent=1)[:4000])
    return rep


if __name__ == "__main__":
    main()
