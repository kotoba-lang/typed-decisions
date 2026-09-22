import json
import os

import torch

from typed_decisions.encoder import DecisionEncoder, Collator, load_tokenizer, MARKERS
from typed_decisions.schema import Question, read_jsonl
from typed_decisions import train_encoder

MODEL = "answerdotai/ModernBERT-base"  # tokenizer only; the model is a tiny random config


def _tiny(tok):
    from transformers import ModernBertConfig, ModernBertModel
    cfg = ModernBertConfig(vocab_size=len(tok) + 8, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=1024, pad_token_id=tok.pad_token_id)
    return DecisionEncoder(ModernBertModel(cfg), 64)


def test_markers_land_on_opt_tokens(synth_data):
    tok = load_tokenizer(MODEL)
    coll = Collator(tok)
    ex = read_jsonl(os.path.join(synth_data, "test.jsonl"))[:5]
    b = coll([(e.state, e.questions) for e in ex])
    m_opt = tok.convert_tokens_to_ids("[OPT]")
    m_q = tok.convert_tokens_to_ids("[Q]")
    assert int((b["input_ids"].gather(1, b["q_pos"].clamp(min=0))[b["q_pos"] >= 0] == m_q).sum()) == sum(len(e.questions) for e in ex)
    got = b["input_ids"].gather(1, b["opt_pos"].clamp(min=0).reshape(5, -1)).reshape(b["opt_pos"].shape)
    n_opt = int(b["opt_mask"].sum())
    assert n_opt == sum(len(q.options) for e in ex for q in e.questions)
    assert int((got[b["opt_mask"]] == m_opt).sum()) == n_opt
    assert int((b["gold"] >= 0).sum()) == sum(len(e.questions) for e in ex)
    # every option's text tokens are in its slot, and question text tokens in the question slot; markers/state/pad in none
    Qm, Om = b["opt_pos"].shape[1:]
    n_opt_tokens = sum(len(coll._ids(o)) for e in ex for q in e.questions for o in q.options)
    n_q_tokens = sum(len(coll._ids(q.instructions)) for e in ex for q in e.questions)
    assert int((b["seg"] < Qm * Om).sum()) == n_opt_tokens
    assert int(((b["seg"] >= Qm * Om) & (b["seg"] < Qm * Om + Qm)).sum()) == n_q_tokens
    assert int((b["input_ids"][b["seg"] < Qm * Om + Qm] == m_opt).sum()) == 0


def test_padding_does_not_leak(synth_data):
    """A short example scored alone and inside a padded batch must give the same logits."""
    torch.manual_seed(0)
    tok = load_tokenizer(MODEL)
    coll = Collator(tok)
    model = _tiny(tok).eval()
    ex = read_jsonl(os.path.join(synth_data, "test.jsonl"))[:3]
    ex[1].state = ex[1].state + " " + "and more words about the box " * 20  # forces padding on the others
    with torch.no_grad():
        alone = coll([(ex[0].state, ex[0].questions)])
        la = model(alone["input_ids"], alone["attention_mask"], alone["opt_pos"], alone["opt_mask"], alone["q_pos"], alone["seg"])
        b = coll([(e.state, e.questions) for e in ex])
        lb = model(b["input_ids"], b["attention_mask"], b["opt_pos"], b["opt_mask"], b["q_pos"], b["seg"])
    la, lb = la[0], lb[0, : la.shape[1], : la.shape[2]]
    finite = torch.isfinite(la)
    diff = (la[finite] - lb[finite]).abs().max().item()
    assert int(finite.sum()) == 4 + 5 + 2
    assert diff < 1e-4, diff


def test_questions_take_priority_when_state_and_augmentation_fill_context():
    tok = load_tokenizer(MODEL)
    coll = Collator(tok, max_state_tokens=512, max_len=64)
    questions = [Question("q", "choice", "Pick the best answer from this deliberately longer instruction.", ["first option", "second option"], 0)]
    ids, positions, _, _ = coll.encode_one("state " * 500, questions)
    assert len(ids) == 64
    assert tok.convert_tokens_to_ids("[OPT]") == ids[positions[0][0]]


def test_end_to_end_learns_rule(synth_data, tmp_path):
    rep = train_encoder.main(["--tiny", "--model", MODEL, "--data", synth_data, "--out", str(tmp_path / "run"),
                              "--epochs", "6", "--batch", "16", "--lr", "2e-3", "--device", "cpu", "--bench-n", "1,10"])
    m = rep["metrics_Tfit"]
    assert m["all"]["n"] == 180
    assert rep["train"]["loss_last10_mean"] < rep["train"]["loss_first"]
    assert m["kind=choice"]["acc"] > 0.9, m["kind=choice"]
    assert m["kind=noul"]["acc"] > 0.9, m["kind=noul"]
    assert m["kind=score"]["score_mae"] < 0.5, m["kind=score"]
    assert rep["latency"][1]["n_questions"] == 10 and rep["latency"][1]["p50_ms"] > 0
    assert os.path.exists(tmp_path / "run" / "report.json")
