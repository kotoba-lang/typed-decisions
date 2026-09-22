"""Backbone A — an encoder (ModernBERT) that answers every question on a state in ONE forward:

    [CLS] [STATE] state [Q] instructions [OPT] opt_1 [OPT] opt_2 ... [Q] instructions [OPT] ... [SEP]

The hidden state at each [OPT] marker is scored by a small head; a softmax *within each
question's option group* is that question's distribution. Nothing is generated, so the
structured-output error rate is 0 by construction and N questions cost one pass over
(state + N question texts) — the Jev shape (typesafe.ai: "single parallel pass").

Loss = cross-entropy + brier_weight * Brier on the same softmax (calibration pressure in-training;
a post-hoc temperature on the validation split is fitted separately and reported as its own row).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import Example, Question

MARKERS = ["[STATE]", "[Q]", "[OPT]"]


class DecisionEncoder(nn.Module):
    """pool = "opt":   logit_j = MLP(h[OPT_j])                                (the option marker alone)
       pool = "q-opt": logit_j = MLP([h[Q]; h[OPT_j]; h[Q] * h[OPT_j]])       (question marker x option marker)
       pool = "span":  same matching head over the MEAN of the option's text tokens and the question's text tokens
    Measured 2026-09-18 (3k states, 1 epoch, ModernBERT-base/large, RoBERTa-base; lr/head-lr/amp/attn/compile
    sweeps): "opt" and "q-opt" stay at the label prior (ce ~1.6) although the loop overfits 16 states to loss 0
    in 50 steps — scoring a fresh marker token's hidden state has no pretrained structure to start from.
    "span" is the default; the README carries the numbers."""

    def __init__(self, backbone, hidden: int, pool: str = "span"):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        inp = hidden if pool == "opt" else 3 * hidden
        self.head = nn.Sequential(nn.Linear(inp, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.temperature = 1.0  # post-hoc, not a Parameter

    @classmethod
    def from_pretrained(cls, name: str, tokenizer, pool: str = "span", **kw):
        from transformers import AutoModel
        bb = AutoModel.from_pretrained(name, **kw)  # kw may carry reference_compile=False (ModernBERT torch.compile of the MLP/attention paths)
        if len(tokenizer) > bb.get_input_embeddings().num_embeddings:
            bb.resize_token_embeddings(len(tokenizer))
        return cls(bb, bb.config.hidden_size, pool)

    def forward(self, input_ids, attention_mask, opt_pos, opt_mask, q_pos=None, seg=None):
        """opt_pos: (B, Qmax, Omax) token positions of [OPT] markers, opt_mask: same shape, bool;
        q_pos: (B, Qmax) positions of the [Q] markers (pool="q-opt"); seg: (B, L) span slot per token (pool="span").
        Returns logits (B, Qmax, Omax) with -inf where opt_mask is False."""
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (B, L, H)
        B, Qm, Om = opt_pos.shape
        H = h.size(-1)
        if self.pool == "span":
            # mean of the *text* tokens of each option and of each question (pretrained-meaningful from step 0,
            # unlike a fresh marker token whose attention pattern has to be learned first)
            n_slots = Qm * Om + Qm
            sums = h.new_zeros((B, n_slots + 1, H)).scatter_add_(1, seg.unsqueeze(-1).expand(-1, -1, H), h)
            cnt = h.new_zeros((B, n_slots + 1)).scatter_add_(1, seg, torch.ones_like(seg, dtype=h.dtype)).clamp(min=1).unsqueeze(-1)
            mean = sums / cnt
            g = mean[:, : Qm * Om].reshape(B, Qm, Om, H)
            q = mean[:, Qm * Om : Qm * Om + Qm].unsqueeze(2).expand(-1, -1, Om, -1)
            g = torch.cat([q, g, q * g], dim=-1)
            return self.head(g).squeeze(-1).masked_fill(~opt_mask, float("-inf"))
        idx = opt_pos.clamp(min=0).reshape(B, Qm * Om)
        g = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, H)).reshape(B, Qm, Om, H)
        if self.pool == "q-opt":
            q = torch.gather(h, 1, q_pos.clamp(min=0).unsqueeze(-1).expand(-1, -1, H))  # (B, Qm, H)
            q = q.unsqueeze(2).expand(-1, -1, Om, -1)
            g = torch.cat([q, g, q * g], dim=-1)
        logits = self.head(g).squeeze(-1)
        return logits.masked_fill(~opt_mask, float("-inf"))


def load_tokenizer(name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.add_special_tokens({"additional_special_tokens": MARKERS})
    return tok


class Collator:
    def __init__(self, tok, max_state_tokens: int = 512, max_len: int = 8192):
        self.tok = tok
        self.max_state = max_state_tokens
        self.max_len = max_len
        self.cls, self.sep = tok.cls_token_id, tok.sep_token_id
        self.m_state, self.m_q, self.m_opt = tok.convert_tokens_to_ids(MARKERS)
        self.OM = 256  # option-segment stride (Jev's max is 255 options)
        self._cache: dict[str, list[int]] = {}

    def _ids(self, text: str) -> list[int]:
        r = self._cache.get(text)
        if r is None:
            r = self.tok(text, add_special_tokens=False)["input_ids"]
            if len(self._cache) < 200_000:
                self._cache[text] = r
        return r

    def encode_one(self, state: str, questions: list[Question]):
        # Questions define the requested typed outputs, so they take precedence over state tail
        # tokens when an augmented instruction makes a pack cross the context limit. Compute the
        # exact question budget first instead of failing late after a mostly completed epoch.
        q_tokens = [(self._ids(q.instructions), [self._ids(o) for o in q.options]) for q in questions]
        question_tokens = sum(1 + len(qt) + sum(1 + len(ot) for ot in opts) for qt, opts in q_tokens)
        state_budget = self.max_len - 3 - question_tokens  # CLS + STATE marker + SEP
        if state_budget < 0:
            raise ValueError(f"questions need {question_tokens + 3} tokens > max_len {self.max_len}: {len(questions)} questions")
        ids = [self.cls, self.m_state] + self._ids(state)[: min(self.max_state, state_budget)]
        positions: list[list[int]] = []
        q_positions: list[int] = []
        spans: list[tuple[int, int, int]] = []  # (segment id, start, end) over token positions; segment = q*OM + o for options, -(q+1) for question text
        for qi, (q, (t, option_tokens)) in enumerate(zip(questions, q_tokens)):
            q_positions.append(len(ids))
            spans.append((-(qi + 1), len(ids) + 1, len(ids) + 1 + len(t)))
            ids += [self.m_q] + t
            pos = []
            for oi, t in enumerate(option_tokens):
                pos.append(len(ids))
                spans.append((qi * self.OM + oi, len(ids) + 1, len(ids) + 1 + len(t)))
                ids += [self.m_opt] + t
            positions.append(pos)
        ids.append(self.sep)
        assert len(ids) <= self.max_len
        return ids, positions, q_positions, spans

    def __call__(self, items: list[tuple[str, list[Question]]], device=None):
        encoded = [self.encode_one(s, qs) for s, qs in items]
        L = max(len(e[0]) for e in encoded)
        Qm = max(len(e[1]) for e in encoded)
        Om = max(len(p) for e in encoded for p in e[1])
        pad = self.tok.pad_token_id
        input_ids = torch.full((len(items), L), pad, dtype=torch.long)
        attn = torch.zeros((len(items), L), dtype=torch.long)
        opt_pos = torch.full((len(items), Qm, Om), -1, dtype=torch.long)
        q_pos = torch.full((len(items), Qm), -1, dtype=torch.long)
        # span pooling targets: token -> slot index; slot = qi*Om + oi for option text, Qm*Om + qi for question text, last = junk
        n_slots = Qm * Om + Qm
        seg = torch.full((len(items), L), n_slots, dtype=torch.long)
        for b, (ids, positions, q_positions, spans) in enumerate(encoded):
            input_ids[b, : len(ids)] = torch.tensor(ids)
            attn[b, : len(ids)] = 1
            q_pos[b, : len(q_positions)] = torch.tensor(q_positions)
            for qi, pos in enumerate(positions):
                opt_pos[b, qi, : len(pos)] = torch.tensor(pos)
            for sid, st, en in spans:
                slot = (Qm * Om + (-sid - 1)) if sid < 0 else ((sid // self.OM) * Om + sid % self.OM)
                seg[b, st:en] = slot
        opt_mask = opt_pos >= 0
        gold = torch.full((len(items), Qm), -100, dtype=torch.long)
        for b, (_, qs) in enumerate(items):
            for qi, q in enumerate(qs):
                gold[b, qi] = q.gold
        out = {"input_ids": input_ids, "attention_mask": attn, "opt_pos": opt_pos, "opt_mask": opt_mask, "q_pos": q_pos, "seg": seg, "gold": gold}
        return {k: (v.to(device) if device is not None else v) for k, v in out.items()}


def decision_loss(logits, gold, brier_weight: float = 1.0):
    """logits (B, Qm, Om) with -inf on padded options; gold (B, Qm) with -100 on padded questions."""
    valid = gold >= 0
    lg = logits[valid]  # (n, Om)
    g = gold[valid]
    ce = F.cross_entropy(lg, g)
    if brier_weight > 0:
        p = lg.softmax(-1)
        onehot = F.one_hot(g, lg.size(-1)).to(p.dtype)
        finite = torch.isfinite(lg)
        brier = (((p - onehot) ** 2) * finite).sum(-1).mean()
    else:
        brier = torch.zeros((), device=lg.device)
    return ce + brier_weight * brier, {"ce": ce.item(), "brier": brier.item(), "n": int(valid.sum())}


@torch.no_grad()
def predict(model: DecisionEncoder, collator: Collator, examples: list[Example], batch_size: int, device, temperature: float | None = None) -> list[dict]:
    """Returns one record per question: {qid, kind, source, probs, gold, logits}."""
    model.eval()
    T = model.temperature if temperature is None else temperature
    out = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        batch = collator([(e.state, e.questions) for e in chunk], device)
        logits = model(batch["input_ids"], batch["attention_mask"], batch["opt_pos"], batch["opt_mask"], batch["q_pos"], batch["seg"]).float()
        probs = (logits / T).softmax(-1)
        for b, e in enumerate(chunk):
            for qi, q in enumerate(e.questions):
                n = len(q.options)
                out.append({"qid": q.qid, "kind": q.kind, "source": e.source, "gold": q.gold,
                            "probs": probs[b, qi, :n].tolist(), "logits": logits[b, qi, :n].tolist()})
    return out
