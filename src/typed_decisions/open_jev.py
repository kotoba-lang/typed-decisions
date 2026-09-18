"""open-jev — load a published typed-decision model from the Hugging Face Hub and decide.

    from typed_decisions.open_jev import OpenJev
    m = OpenJev.from_pretrained("com-kotobalabs/open-jev-deberta-v3-large")
    m.decide("Customer: I was charged twice for the same order.",
             [{"type": "choice", "instructions": "Which team should handle this?", "options": ["billing", "technical", "sales"]},
              {"type": "score",  "instructions": "How frustrated is the customer?", "options": ["calm", "annoyed", "angry"]},
              {"type": "noul",   "instructions": "The customer asks for a refund."}])
    -> [{"choice": "billing", "probabilities": {...}, "confidence": 0.83},
        {"score": 1.2, "probabilities": {...}, "confidence": 0.61},
        {"noul": 0.78}]

One forward pass answers every question; nothing is generated. The bundle is what train_encoder.py
--save writes: the backbone in HF format, `head.safetensors`, the tokenizer with the three marker
tokens, and `open_jev_config.json` (pool, temperature, limits, training provenance, measured metrics).
This file is copied into the model repo so `pip install typed-decisions` is not required to run it.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

from .encoder import DecisionEncoder, Collator
from .schema import Question, NOUL_OPTIONS, readout


class OpenJev:
    def __init__(self, model: DecisionEncoder, tok, collator: Collator, config: dict, device):
        self.model, self.tok, self.collator, self.config, self.device = model, tok, collator, config, device

    @classmethod
    def from_pretrained(cls, repo_or_dir: str, device: str | None = None, revision: str | None = None) -> "OpenJev":
        from transformers import AutoModel, AutoTokenizer
        from safetensors.torch import load_file
        d = repo_or_dir if os.path.isdir(repo_or_dir) else __import__("huggingface_hub").snapshot_download(repo_or_dir, revision=revision)
        cfg = json.load(open(os.path.join(d, "open_jev_config.json")))
        tok = AutoTokenizer.from_pretrained(d)
        bb = AutoModel.from_pretrained(d, attn_implementation=cfg.get("attn_implementation", "eager"))
        m = DecisionEncoder(bb, cfg["hidden"], cfg["pool"])
        m.head.load_state_dict(load_file(os.path.join(d, "head.safetensors")))
        m.temperature = float(cfg.get("temperature", 1.0))
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
        m.to(dev).eval()
        return cls(m, tok, Collator(tok, max_state_tokens=cfg.get("max_state_tokens", 256), max_len=cfg.get("max_len", 512)), cfg, dev)

    @staticmethod
    def _question(i: int, q: dict) -> Question:
        kind = q["type"]
        if kind == "noul":
            return Question(f"q{i}", "noul", q["instructions"], list(NOUL_OPTIONS), 0)
        opts = list(q["options"])
        if kind == "score" and not 2 <= len(opts) <= 10:
            raise ValueError("score takes 2..10 ordered levels")
        if kind == "choice" and not 2 <= len(opts) <= 255:
            raise ValueError("choice takes 2..255 options")
        return Question(f"q{i}", kind, q["instructions"], opts, 0)

    @torch.no_grad()
    def decide(self, state: str, questions: list[dict]) -> list[dict]:
        qs = [self._question(i, q) for i, q in enumerate(questions)]
        b = self.collator([(state, qs)], self.device)
        logits = self.model(b["input_ids"], b["attention_mask"], b["opt_pos"], b["opt_mask"], b["q_pos"], b["seg"]).float()
        probs = (logits / self.model.temperature).softmax(-1)[0]
        out = []
        for qi, q in enumerate(qs):
            p = probs[qi, : len(q.options)].tolist()
            r = readout(q.kind, p)
            if q.kind == "choice":
                out.append({"choice": q.options[r["choice"]], "probabilities": dict(zip(q.options, p)), "confidence": r["confidence"]})
            elif q.kind == "score":
                out.append({"score": r["score"], "probabilities": dict(zip(q.options, p)), "confidence": r["confidence"]})
            else:
                out.append({"noul": r["noul"]})
        return out

    @torch.no_grad()
    def decide_batch(self, items: list[tuple[str, list[dict]]]) -> list[list[dict]]:
        return [self.decide(s, qs) for s, qs in items]
