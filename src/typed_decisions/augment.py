"""Training-time question augmentation with gold preserved by construction, plus the pairs a
consistency loss needs. Measured motivation (ADR-2609181715): in-domain 0.85 -> OOD 0.62; negated
noul answered as the un-negated question by two of three backbones; score does not transfer to
unseen level sets. Every train question always appeared with the same wording and the same option
order, so a slot could be memorised. This module makes that impossible:

  shuffle    permute option order (gold index remapped)             — choice only
  paraphrase rewrite the instructions with a template               — all kinds
  negate     noul: negate the statement, flip the gold              — noul only
  drop       remove up to half of the distractor options            — choice with >= 4 options
  relabel    score: rename the levels with a synonym set of the same length — score only

`augment(q, rng)` returns a new Question with the same qid suffix so predictions still map back.
`pair(q, rng)` returns (q, q') with equal gold for the consistency loss (never `negate`).
"""

from __future__ import annotations

import random

from .schema import Question, NOUL_OPTIONS

PARAPHRASE = {
    "choice": ["{i}", "Decide: {i}", "From the options below, {il}", "Given the state above, {il}", "Pick the best answer. {i}", "Classify the state. {i}"],
    "score": ["{i}", "Rate the state. {i}", "On the scale given, {il}", "Judge: {i}", "Place the state on this scale. {i}"],
    "noul": ["{i}", "Is the following true of the state? {i}", "Claim: {i} Is this correct?", "Does this hold: {i}", "True or false: {i}"],
}
NEGATE = ["It is not the case that: {i}", "The following is FALSE: {i}", "Is it wrong to say: {i}", "Contrary to the state: {i} (is this false?)"]
SCORE_SYNONYMS = {
    3: [["low", "medium", "high"], ["weak", "moderate", "strong"], ["barely", "somewhat", "very much"]],
    5: [["lowest", "low", "middle", "high", "highest"], ["strongly negative", "somewhat negative", "neutral", "somewhat positive", "strongly positive"], ["1 of 5", "2 of 5", "3 of 5", "4 of 5", "5 of 5"]],
}


def _lower_first(s: str) -> str:
    return s[0].lower() + s[1:] if s and s[0].isupper() and not s.startswith("I ") else s


def paraphrase(q: Question, rng: random.Random) -> Question:
    t = rng.choice(PARAPHRASE[q.kind])
    return Question(q.qid, q.kind, t.format(i=q.instructions, il=_lower_first(q.instructions)), list(q.options), q.gold)


def shuffle(q: Question, rng: random.Random) -> Question:
    if q.kind != "choice":  # score levels are ordered; noul's (no, yes) is fixed by the schema
        return q
    idx = list(range(len(q.options)))
    rng.shuffle(idx)
    return Question(q.qid, q.kind, q.instructions, [q.options[i] for i in idx], idx.index(q.gold))


def negate(q: Question, rng: random.Random) -> Question:
    if q.kind != "noul":
        return q
    return Question(q.qid, "noul", rng.choice(NEGATE).format(i=q.instructions), list(NOUL_OPTIONS), 1 - q.gold)


def drop(q: Question, rng: random.Random) -> Question:
    if q.kind != "choice" or len(q.options) < 4:
        return q
    distractors = [i for i in range(len(q.options)) if i != q.gold]
    keep = sorted(rng.sample(distractors, max(2, len(distractors) - rng.randint(1, len(distractors) // 2))) + [q.gold])
    return Question(q.qid, "choice", q.instructions, [q.options[i] for i in keep], keep.index(q.gold))


def relabel(q: Question, rng: random.Random) -> Question:
    if q.kind != "score" or len(q.options) not in SCORE_SYNONYMS:
        return q
    return Question(q.qid, "score", q.instructions, list(rng.choice(SCORE_SYNONYMS[len(q.options)])), q.gold)


OPS = [shuffle, paraphrase, drop, relabel]


def augment(q: Question, rng: random.Random, p_negate: float = 0.3) -> Question:
    out = q
    for op in OPS:
        if rng.random() < 0.5:
            out = op(out, rng)
    if q.kind == "noul" and rng.random() < p_negate:
        out = negate(out, rng)
    return out


def pair(q: Question, rng: random.Random) -> tuple[Question, Question]:
    """Two surface forms of the same decision (same gold), for a consistency loss."""
    a = q
    b = q
    for op in (shuffle, paraphrase, drop, relabel):
        if rng.random() < 0.7:
            b = op(b, rng)
    if b.instructions == a.instructions and b.options == a.options:
        b = paraphrase(shuffle(a, rng), rng)
    return a, b
