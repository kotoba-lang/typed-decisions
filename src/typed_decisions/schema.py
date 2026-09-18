"""The decision schema, shaped after Jev's public API (typesafe.ai, dev.to guide 2026-09):

    state + questions -> answers, each with a probability distribution and a confidence.

Three question kinds (names kept as Jev names them so the report reads against their page):
  choice  one of N unordered options              -> choice, probabilities, confidence
  score   one of N *ordered* levels (2..10)        -> score (expected level, may fall between levels), probabilities, confidence
  noul    yes/no                                   -> noul = p(yes)

Every kind is the same thing to a model: a softmax over an option list. `score` differs only in
how the distribution is read out; `noul` is a 2-option choice whose read-out is p(options[1]).
That is the whole point of the Jev shape and why one head serves all three.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal
import json
import math

Kind = Literal["choice", "score", "noul"]

NOUL_OPTIONS = ("no", "yes")


@dataclass
class Question:
    qid: str
    kind: Kind
    instructions: str
    options: list[str]
    gold: int  # index into options

    def __post_init__(self):
        if self.kind == "noul":
            assert tuple(self.options) == NOUL_OPTIONS, "noul options are fixed (no, yes)"
        if self.kind == "score":
            assert 2 <= len(self.options) <= 10, "score has 2..10 ordered levels"
        assert 0 <= self.gold < len(self.options), (self.qid, self.gold, len(self.options))


@dataclass
class Example:
    state: str
    questions: list[Question]
    source: str
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"state": self.state, "source": self.source, "meta": self.meta,
                           "questions": [asdict(q) for q in self.questions]}, ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "Example":
        d = json.loads(line)
        return Example(state=d["state"], source=d["source"], meta=d.get("meta", {}),
                       questions=[Question(**q) for q in d["questions"]])


def read_jsonl(path) -> list[Example]:
    with open(path) as f:
        return [Example.from_json(l) for l in f if l.strip()]


def write_jsonl(path, examples: list[Example]) -> None:
    with open(path, "w") as f:
        for e in examples:
            f.write(e.to_json() + "\n")


def readout(kind: Kind, probs: list[float]) -> dict:
    """Turn a distribution over options into the Jev-shaped answer for that kind."""
    k = max(range(len(probs)), key=lambda i: probs[i])
    conf = float(probs[k])
    if kind == "choice":
        return {"choice": k, "probabilities": probs, "confidence": conf}
    if kind == "score":
        return {"score": float(sum(i * p for i, p in enumerate(probs))), "probabilities": probs, "confidence": conf}
    if kind == "noul":
        return {"noul": float(probs[1])}
    raise ValueError(kind)


def entropy_confidence(probs: list[float]) -> float:
    """1 - H(p)/log(N): an alternative confidence that is 1 at a point mass and 0 at uniform."""
    n = len(probs)
    if n < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return 1.0 - h / math.log(n)
