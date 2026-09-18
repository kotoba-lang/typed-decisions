import os

import torch

from typed_decisions.dllm import DllmDecider, OptionAlphabet
from typed_decisions.schema import read_jsonl
from typed_decisions import train_dllm

MODEL = "answerdotai/ModernBERT-base"  # any tokenizer; model is a tiny random Llama in --tiny mode


def test_alphabet_and_slots(synth_data):
    tok, model = train_dllm.load(MODEL, torch.device("cpu"), torch.float32, tiny=True)
    alpha = OptionAlphabet(tok)
    assert len(alpha) >= 77, len(alpha)  # banking77 needs 77 labels; Jev allows 255
    assert len(set(alpha.ids)) == len(alpha.ids)
    mask_id = DllmDecider.resolve_mask_id(tok, model)
    dec = DllmDecider(model, tok, alpha, mask_id)
    ex = read_jsonl(os.path.join(synth_data, "test.jsonl"))[:2]
    ids, slots, opt_ids = dec.encode_one(ex[0].state, ex[0].questions)
    assert [ids[s] for s in slots] == [mask_id] * 3
    assert [len(o) for o in opt_ids] == [4, 5, 2]
    probs = dec.decide([(e.state, e.questions) for e in ex], torch.device("cpu"), steps=1)
    assert [len(p) for p in probs[0]] == [4, 5, 2]
    assert all(abs(sum(p) - 1) < 1e-5 for row in probs for p in row)
    probs3 = dec.decide([(e.state, e.questions) for e in ex], torch.device("cpu"), steps=3)
    assert all(p is not None for row in probs3 for p in row)


def test_end_to_end_learns_rule(synth_data, tmp_path):
    rep = train_dllm.main(["--tiny", "--no-lora", "--model", MODEL, "--data", synth_data, "--out", str(tmp_path / "run"),
                           "--epochs", "8", "--batch", "16", "--grad-accum", "1", "--lr", "2e-3", "--device", "cpu",
                           "--bench-n", "1,10", "--eval-steps", "1,2", "--skip-zero-shot"])
    m = rep["metrics_trained_steps1_Tfit"]
    assert m["all"]["n"] == 180
    assert rep["train"]["loss_last10_mean"] < rep["train"]["loss_first"]
    assert m["kind=choice"]["acc"] > 0.9, m["kind=choice"]
    assert m["kind=noul"]["acc"] > 0.9, m["kind=noul"]
    assert "metrics_trained_steps2_T1" in rep
    assert any(l["steps"] == 2 for l in rep["latency"])
