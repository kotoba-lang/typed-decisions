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

import argparse
import json
import os
import sys

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


def decide_request(model: OpenJev, request: dict) -> dict:
    """Validate one JSON wire request and return a provenance-bearing result.

    This is deliberately a typed-decision surface, not a text-generation
    compatibility endpoint.  A caller supplies a closed option set and gets
    one calibrated distribution per question; unknown request fields and
    malformed questions fail before a model forward.
    """
    if set(request) != {"state", "questions"}:
        raise ValueError("request must contain exactly state and questions")
    if not isinstance(request["state"], str) or not request["state"]:
        raise ValueError("state must be a non-empty string")
    if not isinstance(request["questions"], list) or not request["questions"]:
        raise ValueError("questions must be a non-empty list")
    for index, question in enumerate(request["questions"]):
        if not isinstance(question, dict):
            raise ValueError(f"question {index} must be an object")
        kind = question.get("type")
        expected = {"type", "instructions"} if kind == "noul" else {"type", "instructions", "options"}
        if kind not in {"choice", "score", "noul"} or set(question) != expected:
            raise ValueError(f"question {index} has an invalid kind or field set")
        if not isinstance(question["instructions"], str) or not question["instructions"]:
            raise ValueError(f"question {index} instructions must be a non-empty string")
        if kind != "noul":
            options = question["options"]
            limit = 10 if kind == "score" else 255
            if (not isinstance(options, list) or not 2 <= len(options) <= limit or
                    not all(isinstance(option, str) and option for option in options) or
                    len(set(options)) != len(options)):
                raise ValueError(f"question {index} options are invalid")
    decisions = model.decide(request["state"], request["questions"])
    return {
        "kind": "typed-decisions/open-jev-v1",
        "generated_text": False,
        "model": {
            "base_model": model.config.get("base_model"),
            "pool": model.config.get("pool"),
            "temperature": model.config.get("temperature"),
            "train": model.config.get("train"),
            "metrics": model.config.get("metrics"),
        },
        "decisions": decisions,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a trained OpenJev typed-decision model over one JSON request from stdin."
    )
    parser.add_argument("--model", default="com-kotobalabs/open-jev-deberta-v3-large")
    parser.add_argument("--revision")
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    try:
        request = json.load(sys.stdin)
        model = OpenJev.from_pretrained(args.model, device=args.device, revision=args.revision)
        result = decide_request(model, request)
        result["artifact"] = {"repo_or_dir": args.model, "revision": args.revision}
        json.dump(result, sys.stdout, ensure_ascii=False, separators=(",", ":"))
        sys.stdout.write("\n")
        return 0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": "invalid-request", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 64


if __name__ == "__main__":
    raise SystemExit(main())
