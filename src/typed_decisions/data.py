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
        out.append(Example(state=r["text"], source="banking77", meta={"i": i, "split": split}, questions=[
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


SOURCES = {
    "banking77": ("train", "test"),
    "sst5": ("train", "test"),
    "boolq": ("train", "validation"),
}
LOADERS = {"banking77": _banking77, "sst5": _sst5, "boolq": _boolq}


def build(out_dir: str, n_train: int, n_val: int, n_test: int, seed: int = 0) -> dict:
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
    rng.shuffle(train)
    rng.shuffle(test)
    write_jsonl(os.path.join(out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(out_dir, "val.jsonl"), val)
    write_jsonl(os.path.join(out_dir, "test.jsonl"), test)
    counts["questions"] = {k: sum(len(e.questions) for e in v) for k, v in (("train", train), ("val", val), ("test", test))}
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import json
    print(json.dumps(build(a.out, a.n_train, a.n_val, a.n_test, a.seed), indent=1))


if __name__ == "__main__":
    main()
