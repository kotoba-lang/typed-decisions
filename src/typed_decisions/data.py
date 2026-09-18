"""Build the decision corpus from three public datasets, each contributing the question kinds it
can answer with *gold* labels (no synthetic answers, no teacher — see README "What the labels are"):

  mteb/banking77     state = customer message (PolyAI/banking77 data, parquet mirror)
      choice  intent (77 options, Jev allows up to 255)
      choice  product area (10 options, a deterministic keyword rule over the 77 intent names)
      noul    "asks about a card" (intent name contains `card`)
  SetFit/sst5        state = review sentence
      score   sentiment level (5 ordered levels)
      choice  polarity (negative / neutral / positive, from the level)
      noul    "expresses a positive opinion" (level >= 3)
  google/boolq       state = passage
      noul    the dataset's own question

Every question on a state is answerable from that state, so a state carries 1..3 questions and a
single forward pass answers all of them — the shape the latency benchmark packs further.
"""

from __future__ import annotations

import argparse
import os
import random

from .schema import Example, Question, NOUL_OPTIONS, write_jsonl

AREA_RULES = [  # first match wins; the rule is the label, not a claim about banking
    ("fees & charges", ("fee", "charge")),
    ("pin & security", ("pin", "passcode", "compromised")),
    ("refund & dispute", ("refund", "dispute", "not_recognised", "reverted", "wrong_amount")),
    ("top-up", ("top_up", "topping_up")),
    ("exchange & fiat", ("exchange", "fiat")),
    ("atm & cash", ("atm", "cash")),
    ("transfer", ("transfer", "beneficiary", "receiving_money")),
    ("card", ("card",)),
    ("account & identity", ("account", "identity", "verify", "age", "country", "edit_personal")),
]
AREAS = [a for a, _ in AREA_RULES] + ["other"]


def banking_area(label_name: str) -> int:
    n = label_name.lower()
    for i, (_, kws) in enumerate(AREA_RULES):
        if any(k in n for k in kws):
            return i
    return len(AREA_RULES)


def _banking77(split: str) -> list[Example]:
    from datasets import load_dataset
    ds = load_dataset("mteb/banking77", split=split)  # PolyAI/banking77 is a script dataset (unsupported by datasets>=4)
    names = [None] * 77
    for r in ds:
        names[int(r["label"])] = r["label_text"]
    assert all(names), "label_text missing for some label"
    opts = [n.replace("_", " ") for n in names]
    out = []
    for i, r in enumerate(ds):
        lab = int(r["label"])
        out.append(Example(state=r["text"], source="banking77", meta={"i": i, "split": split, "label_name": names[lab]}, questions=[
            Question(f"b77-{split}-{i}-intent", "choice", "Which banking intent does the customer message express?", opts, lab),
            Question(f"b77-{split}-{i}-area", "choice", "Which product area is the message about?", AREAS, banking_area(names[lab])),
            Question(f"b77-{split}-{i}-card", "noul", "The customer is asking about a card (physical or virtual).", list(NOUL_OPTIONS), int("card" in names[lab].lower())),
        ]))
    return out


SST_LEVELS = ["very negative", "negative", "neutral", "positive", "very positive"]


def _sst5(split: str) -> list[Example]:
    from datasets import load_dataset
    ds = load_dataset("SetFit/sst5", split=split)
    out = []
    for i, r in enumerate(ds):
        lab = int(r["label"])
        pol = 0 if lab < 2 else (1 if lab == 2 else 2)
        out.append(Example(state=r["text"], source="sst5", meta={"i": i, "split": split}, questions=[
            Question(f"sst5-{split}-{i}-level", "score", "How positive is the sentiment of this review sentence?", SST_LEVELS, lab),
            Question(f"sst5-{split}-{i}-polarity", "choice", "What is the overall polarity of the sentence?", ["negative", "neutral", "positive"], pol),
            Question(f"sst5-{split}-{i}-positive", "noul", "The sentence expresses a positive opinion.", list(NOUL_OPTIONS), int(lab >= 3)),
        ]))
    return out


def _boolq(split: str) -> list[Example]:
    from datasets import load_dataset
    ds = load_dataset("google/boolq", split=split)
    out = []
    for i, r in enumerate(ds):
        q = r["question"].strip()
        q = q[0].upper() + q[1:] + ("?" if not q.endswith("?") else "")
        out.append(Example(state=r["passage"], source="boolq", meta={"i": i, "split": split}, questions=[
            Question(f"boolq-{split}-{i}", "noul", q, list(NOUL_OPTIONS), int(bool(r["answer"]))),
        ]))
    return out


# ---- OOD questions: never seen in training (new instructions AND new option sets on the same test
# states). Gold is derived from the same dataset labels by a deterministic rule, so the split tests
# whether the model reads the question, not whether it memorised a slot.
OOD_B77_TOPICS = [  # which of these does the intent belong to? (a different partition than AREAS)
    ("money going out of the account", ("transfer", "payment", "withdraw", "cash", "direct_debit", "purchase")),
    ("money coming into the account", ("top_up", "topping_up", "receiving", "refund", "reverted", "deposit")),
    ("the card as a physical or virtual object", ("card_", "_card", "virtual_card", "contactless", "pin")),
    ("account setup, limits and identity", ("account", "identity", "verify", "age", "country", "edit_personal", "why_verify")),
]
OOD_B77_TOPIC_NAMES = [t for t, _ in OOD_B77_TOPICS] + ["none of these"]


def b77_topic(label_name: str) -> int:
    n = label_name.lower()
    for i, (_, kws) in enumerate(OOD_B77_TOPICS):
        if any(k in n for k in kws):
            return i
    return len(OOD_B77_TOPICS)


def ood_questions(e: Example, label_name: str | None = None) -> list[Question]:
    """OOD questions for one test example; `label_name` is the banking77 intent name when source is banking77."""
    i = e.meta.get("i")
    split = e.meta.get("split")
    if e.source == "banking77":
        lab = label_name
        return [
            Question(f"ood-b77-{split}-{i}-topic", "choice", "Pick the topic that best describes what the customer's message is about.", OOD_B77_TOPIC_NAMES, b77_topic(lab)),
            Question(f"ood-b77-{split}-{i}-outflow", "noul", "Is the customer talking about money leaving their account (a payment, transfer, withdrawal or purchase)?", list(NOUL_OPTIONS), int(b77_topic(lab) == 0)),
            # no ordered (score) OOD question for banking77: the dataset has no ordinal ground truth, and the
            # "urgency" rule used on 2026-09-18 was answered below majority by three models AND the teacher
            # (0.31) — a label problem, not a model problem. Removed (ADR-2609181715 §4).
        ]
    if e.source == "sst5":
        lev = e.questions[0].gold  # 0..4
        return [
            Question(f"ood-sst5-{split}-{i}-recommend", "noul", "Would the reviewer recommend this to a friend?", list(NOUL_OPTIONS), int(lev >= 3)),
            Question(f"ood-sst5-{split}-{i}-tone", "choice", "Which word best describes the reviewer's tone?", ["dismissive", "lukewarm", "enthusiastic"], 0 if lev <= 1 else (1 if lev == 2 else 2)),
            # ordered questions whose gold is a MONOTONE relabelling of the dataset's own 5 levels — unseen
            # option text and instructions, defensible gold. `stars` keeps the direction, `disappointed`
            # reverses it (reads the scale, not the slot). The former "intensity" (|level-2|) question was
            # below majority for three models and the teacher alike (0.39) and is removed.
            Question(f"ood-sst5-{split}-{i}-stars", "score", "How many stars out of five would the reviewer give?", ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"], lev),
            Question(f"ood-sst5-{split}-{i}-disappointed", "score", "How disappointed does the reviewer sound?", ["not at all", "slightly", "somewhat", "quite", "very"], 4 - lev),
        ]
    if e.source == "boolq":
        q = e.questions[0]
        return [
            Question(f"ood-boolq-{split}-{i}-negated", "noul", "Consider the claim: " + q.instructions.rstrip("?") + ". Is this claim FALSE according to the passage?", list(NOUL_OPTIONS), 1 - q.gold),
            Question(f"ood-boolq-{split}-{i}-support", "choice", "According to the passage, the statement '" + q.instructions.rstrip("?") + "' is:", ["supported", "contradicted"], 0 if q.gold == 1 else 1),
        ]
    return []


# ---- extra TRAIN question families on the same states (not the OOD test families): tests whether
# question-side diversity on the same state — rather than more domains — moves OOD. Rule gold as above.
HOWTO_KW = ("why", "how", "what", "can_", "supported", "limit", "age", "country", "estimate", "fee", "rate", "when", "where", "which")
PROBLEM_KW = ("lost", "stolen", "compromised", "not_working", "declined", "failed", "wrong", "not_recognised", "dispute", "reverted", "pending", "not_updated", "not_arrived", "unable", "problem", "error")


def train_families(e: Example, label_name: str | None = None) -> list[Question]:
    i, split = e.meta.get("i"), e.meta.get("split")
    if e.source == "banking77" and label_name:
        n = label_name.lower()
        howto = any(k in n for k in HOWTO_KW) and not any(k in n for k in PROBLEM_KW)
        problem = any(k in n for k in PROBLEM_KW)
        kind = 0 if problem else (1 if howto else 2)
        return [
            Question(f"tf-b77-{split}-{i}-kind", "choice", "What kind of message is this?", ["a report of something that went wrong", "a question about how something works or what is allowed", "a request to do or change something"], kind),
            Question(f"tf-b77-{split}-{i}-problem", "noul", "Is the customer reporting a problem that has already happened?", list(NOUL_OPTIONS), int(problem)),
            Question(f"tf-b77-{split}-{i}-severity", "score", "How serious is the situation described?", ["routine", "needs attention", "urgent"], 2 if any(k in n for k in ("lost", "stolen", "compromised", "not_recognised", "dispute")) else (1 if problem else 0)),
        ]
    if e.source == "sst5":
        lev = e.questions[0].gold
        return [
            Question(f"tf-sst5-{split}-{i}-again", "score", "How likely is the reviewer to watch this again?", ["never", "unlikely", "maybe", "likely", "certainly"], lev),
            Question(f"tf-sst5-{split}-{i}-mixed", "noul", "Is the review mixed or neutral rather than clearly positive or negative?", list(NOUL_OPTIONS), int(lev == 2)),
            Question(f"tf-sst5-{split}-{i}-thumbs", "choice", "Thumbs up, thumbs down, or neither?", ["thumbs down", "neither", "thumbs up"], 0 if lev < 2 else (1 if lev == 2 else 2)),
        ]
    if e.source == "boolq":
        q = e.questions[0]
        return [
            Question(f"tf-boolq-{split}-{i}-answer", "choice", "According to the passage, what is the answer to: " + q.instructions, ["no", "yes"], q.gold),
            Question(f"tf-boolq-{split}-{i}-wrong", "noul", "Would answering 'yes' to the question be wrong given the passage? Question: " + q.instructions, list(NOUL_OPTIONS), 1 - q.gold),
        ]
    return []


SOURCES = {
    "banking77": ("train", "test"),
    "sst5": ("train", "test"),
    "boolq": ("train", "validation"),
}
LOADERS = {"banking77": _banking77, "sst5": _sst5, "boolq": _boolq}


def build(out_dir: str, n_train: int, n_val: int, n_test: int, seed: int = 0, extra_train_families: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    train, val, test = [], [], []
    counts = {}
    for src, (tr_split, te_split) in SOURCES.items():
        tr = LOADERS[src](tr_split)
        te = LOADERS[src](te_split)
        rng.shuffle(tr)
        rng.shuffle(te)
        v, t = tr[:n_val], tr[n_val:n_val + n_train]
        val += v
        train += t
        test += te[:n_test]
        counts[src] = {"train": len(t), "val": len(v), "test": len(te[:n_test]),
                       "available_train": len(tr), "available_test": len(te)}
    if extra_train_families:
        # a second example per train state carrying the extra families (kept separate so the original
        # example's packing, and therefore its tokens, is unchanged)
        train += [Example(state=e.state, source=e.source, meta=dict(e.meta, families=True), questions=train_families(e, e.meta.get("label_name"))) for e in list(train)]
        train = [e for e in train if e.questions]
    rng.shuffle(train)
    rng.shuffle(test)
    write_jsonl(os.path.join(out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(out_dir, "val.jsonl"), val)
    write_jsonl(os.path.join(out_dir, "test.jsonl"), test)
    ood = [Example(state=e.state, source=e.source, meta=dict(e.meta, ood=True), questions=ood_questions(e, e.meta.get("label_name"))) for e in test]
    ood = [e for e in ood if e.questions]
    write_jsonl(os.path.join(out_dir, "ood-test.jsonl"), ood)
    counts["ood_test"] = {"states": len(ood), "questions": sum(len(e.questions) for e in ood)}
    counts["questions"] = {k: sum(len(e.questions) for e in v) for k, v in (("train", train), ("val", val), ("test", test))}
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-train-families", action="store_true")
    a = ap.parse_args(argv)
    import json
    print(json.dumps(build(a.out, a.n_train, a.n_val, a.n_test, a.seed, a.extra_train_families), indent=1))


if __name__ == "__main__":
    main()
