"""Backbone B — a masked-diffusion LM (LLaDA family) with one [MASK] *decision slot* per question:

    <state>...</state>
    Q1. instructions   Options: A) ...  B) ...
    Q2. ...
    A1: [MASK]
    A2: [MASK]

One forward denoises every slot at once (steps = 1, the Jev shape); with steps > 1 the most
confident slots are committed and the rest re-denoised (LLaDA's low-confidence remasking), so the
same model also measures what each extra diffusion step costs. Each slot's distribution is the
softmax of the slot's logits restricted to the *option label tokens* of that question, so the
structured-output error rate is 0 here too, and the loss is the same restricted CE (+ Brier) the
encoder trains with — the comparison is backbone vs backbone, not objective vs objective.

Option labels are single tokens chosen from the tokenizer at load time (`OptionAlphabet`), so a
question with N options costs N label tokens and one slot, never a multi-token answer.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .schema import Example, Question


class OptionAlphabet:
    """Single-token option labels discovered from the tokenizer, so `n_options <= len(alphabet)`."""

    CANDIDATES = ([f" {c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"] + [f" {c}" for c in "abcdefghijklmnopqrstuvwxyz"]
                  + [f" {i}" for i in range(0, 300)] + [f" {a}{b}" for a in "ABCDEFGH" for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])

    def __init__(self, tok, need: int = 255):
        self.labels: list[str] = []
        self.ids: list[int] = []
        seen = set()
        for s in self.CANDIDATES:
            ids = tok(s, add_special_tokens=False)["input_ids"]
            if len(ids) == 1 and ids[0] not in seen:
                seen.add(ids[0])
                self.labels.append(s)
                self.ids.append(ids[0])
            if len(self.ids) >= need:
                break
        if len(self.ids) < 2:
            raise RuntimeError("REFUSE: tokenizer yields < 2 single-token option labels")

    def __len__(self):
        return len(self.ids)


class DllmDecider:
    def __init__(self, model, tok, alphabet: OptionAlphabet, mask_id: int, max_state_tokens: int = 512):
        self.model, self.tok, self.alphabet, self.mask_id = model, tok, alphabet, mask_id
        self.max_state = max_state_tokens
        self.newline = tok("\n", add_special_tokens=False)["input_ids"]
        self._cache: dict[str, list[int]] = {}

    @staticmethod
    def resolve_mask_id(tok, model) -> int:
        for cand in (getattr(tok, "mask_token_id", None), getattr(model.config, "mask_token_id", None)):
            if isinstance(cand, int) and cand >= 0:
                return cand
        for t in ("<|mdm_mask|>", "[MASK]", "<mask>"):
            i = tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
                return i
        raise RuntimeError("REFUSE: no mask token id on tokenizer/config")

    def _ids(self, text: str) -> list[int]:
        r = self._cache.get(text)
        if r is None:
            r = self.tok(text, add_special_tokens=False)["input_ids"]
            if len(self._cache) < 200_000:
                self._cache[text] = r
        return r

    def encode_one(self, state: str, questions: list[Question]):
        """Returns (ids, slot_positions, option_token_ids per question)."""
        if any(len(q.options) > len(self.alphabet) for q in questions):
            raise ValueError(f"a question has more options than the alphabet ({len(self.alphabet)})")
        bos = [self.tok.bos_token_id] if self.tok.bos_token_id is not None else []
        ids = bos + self._ids("<state>\n") + self._ids(state)[: self.max_state] + self._ids("\n</state>\nAnswer each question with the label of one option.\n")
        for qi, q in enumerate(questions):
            opts = "  ".join(f"{self.alphabet.labels[j].strip()}) {o}" for j, o in enumerate(q.options))
            ids += self._ids(f"Q{qi + 1}. {q.instructions}\nOptions: {opts}\n")
        slots, opt_ids = [], []
        for qi, q in enumerate(questions):
            ids += self._ids(f"A{qi + 1}:")
            slots.append(len(ids))
            ids.append(self.mask_id)
            ids += self.newline
            opt_ids.append(self.alphabet.ids[: len(q.options)])
        return ids, slots, opt_ids

    def collate(self, items: list[tuple[str, list[Question]]], device):
        enc = [self.encode_one(s, qs) for s, qs in items]
        L = max(len(e[0]) for e in enc)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else (self.tok.eos_token_id or 0)
        input_ids = torch.full((len(items), L), pad, dtype=torch.long)
        attn = torch.zeros((len(items), L), dtype=torch.long)
        for b, (ids, _, _) in enumerate(enc):
            input_ids[b, : len(ids)] = torch.tensor(ids)
            attn[b, : len(ids)] = 1
        return input_ids.to(device), attn.to(device), [e[1] for e in enc], [e[2] for e in enc]

    def slot_logits(self, input_ids, attn, slots, opt_ids):
        """Full forward; returns per (b, q) the logits restricted to that question's option tokens."""
        out = self.model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits if hasattr(out, "logits") else out[0]
        res = []
        for b in range(input_ids.size(0)):
            row = []
            for pos, oids in zip(slots[b], opt_ids[b]):
                row.append(logits[b, pos, oids].float())
            res.append(row)
        return res

    def loss(self, input_ids, attn, slots, opt_ids, golds, brier_weight: float = 1.0):
        res = self.slot_logits(input_ids, attn, slots, opt_ids)
        ce, brier, n = 0.0, 0.0, 0
        for b, row in enumerate(res):
            for q, lg in enumerate(row):
                g = golds[b][q]
                ce = ce + F.cross_entropy(lg.unsqueeze(0), torch.tensor([g], device=lg.device))
                p = lg.softmax(-1)
                onehot = F.one_hot(torch.tensor(g, device=lg.device), lg.numel()).to(p.dtype)
                brier = brier + ((p - onehot) ** 2).sum()
                n += 1
        ce, brier = ce / n, brier / n
        return ce + brier_weight * brier, {"ce": float(ce.detach()), "brier": float(brier.detach()), "n": n}

    @torch.no_grad()
    def decide(self, items: list[tuple[str, list[Question]]], device, steps: int = 1, temperature: float = 1.0):
        """Returns per item a list of prob vectors (one per question). `steps` diffusion steps:
        each step commits the ceil(n/steps) most confident still-masked slots."""
        input_ids, attn, slots, opt_ids = self.collate(items, device)
        B = input_ids.size(0)
        probs_out = [[None] * len(slots[b]) for b in range(B)]
        remaining = [set(range(len(slots[b]))) for b in range(B)]
        per_step = [max(1, math.ceil(len(slots[b]) / steps)) for b in range(B)]
        for _ in range(steps):
            if all(not r for r in remaining):
                break
            res = self.slot_logits(input_ids, attn, slots, opt_ids)
            for b in range(B):
                cand = []
                for q in remaining[b]:
                    p = (res[b][q] / temperature).softmax(-1)
                    cand.append((float(p.max()), q, p))
                cand.sort(key=lambda x: -x[0])
                for _, q, p in cand[: per_step[b]]:
                    probs_out[b][q] = p.tolist()
                    input_ids[b, slots[b][q]] = opt_ids[b][q][int(p.argmax())]
                    remaining[b].discard(q)
        return probs_out


@torch.no_grad()
def predict(decider: DllmDecider, examples: list[Example], batch_size: int, device, steps: int = 1, temperature: float = 1.0) -> list[dict]:
    out = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        probs = decider.decide([(e.state, e.questions) for e in chunk], device, steps=steps, temperature=temperature)
        for b, e in enumerate(chunk):
            for qi, q in enumerate(e.questions):
                p = probs[b][qi]
                out.append({"qid": q.qid, "kind": q.kind, "source": e.source, "gold": q.gold, "probs": p,
                            "logits": [math.log(max(x, 1e-12)) for x in p]})
    return out


def find_lora_targets(model) -> list[str]:
    """Attention/MLP projection names present in this (remote-code) model, for peft."""
    import torch.nn as nn
    wanted = {"q_proj", "k_proj", "v_proj", "o_proj", "att_proj", "attn_out", "ff_proj", "ff_out", "gate_proj", "up_proj", "down_proj", "qkv_proj", "query_key_value", "dense"}
    found = set()
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in wanted:
                found.add(leaf)
    return sorted(found)
