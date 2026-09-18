"""Commit 1 of the OOD plan (ADR-2609181715 follow-up): breadth, not depth. Thirteen more public datasets,
each turned into 1–3 typed questions with gold from the dataset's own label, so training sees many
question FAMILIES (NLI, paraphrase, acceptability, emotion, intent, topic, star rating, counterfactual,
review polarity) instead of more states of the same three. The original corpus (`data/`) and its OOD
test are kept byte-identical so the OOD number stays comparable; each new source also contributes a
small in-domain test slice.

    python -m typed_decisions.data_multi --base data --out data-multi --n-train 3000 --n-test 300
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil

from .schema import Example, Question, NOUL_OPTIONS, read_jsonl, write_jsonl

EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
AG = ["world news", "sports", "business", "science and technology"]
DBP = ["company", "educational institution", "artist", "athlete", "office holder", "means of transportation", "building", "natural place", "village", "animal", "plant", "album", "film", "written work"]
STARS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]


def _pair(a: str, b: str, la: str = "Text A", lb: str = "Text B") -> str:
    return f"{la}: {a}\n{lb}: {b}"


def build_source(name: str, split: str, n: int, rng: random.Random) -> list[Example]:
    from datasets import load_dataset
    out = []

    def add(i, state, qs, extra=None):
        out.append(Example(state=state, source=name, meta=dict({"i": i, "split": split}, **(extra or {})), questions=qs))

    if name == "mnli":
        ds = load_dataset("nyu-mll/glue", "mnli", split=split)
        idx = rng.sample(range(len(ds)), min(n, len(ds)))
        for i in idx:
            r = ds[i]; lab = int(r["label"])
            if lab < 0: continue
            add(i, _pair(r["premise"], r["hypothesis"], "Premise", "Hypothesis"), [
                Question(f"mnli-{split}-{i}-rel", "choice", "How does the hypothesis relate to the premise?", ["it follows from the premise", "it is undetermined by the premise", "it contradicts the premise"], lab),
                Question(f"mnli-{split}-{i}-ent", "noul", "Does the premise entail the hypothesis?", list(NOUL_OPTIONS), int(lab == 0))])
    elif name == "rte":
        ds = load_dataset("nyu-mll/glue", "rte", split=split)
        for i in rng.sample(range(len(ds)), min(n, len(ds))):
            r = ds[i]
            add(i, _pair(r["sentence1"], r["sentence2"], "Text", "Claim"), [Question(f"rte-{split}-{i}", "noul", "Is the claim supported by the text?", list(NOUL_OPTIONS), int(r["label"] == 0))])
    elif name == "qnli":
        ds = load_dataset("nyu-mll/glue", "qnli", split=split)
        for i in rng.sample(range(len(ds)), min(n, len(ds))):
            r = ds[i]
            add(i, _pair(r["question"], r["sentence"], "Question", "Sentence"), [Question(f"qnli-{split}-{i}", "noul", "Does the sentence contain the answer to the question?", list(NOUL_OPTIONS), int(r["label"] == 0))])
    elif name == "mrpc":
        ds = load_dataset("nyu-mll/glue", "mrpc", split=split)
        for i in rng.sample(range(len(ds)), min(n, len(ds))):
            r = ds[i]
            add(i, _pair(r["sentence1"], r["sentence2"]), [Question(f"mrpc-{split}-{i}", "noul", "Do the two sentences say the same thing?", list(NOUL_OPTIONS), int(r["label"] == 1))])
    elif name == "paws":
        ds = load_dataset("google-research-datasets/paws", "labeled_final", split=split)
        for i in rng.sample(range(len(ds)), min(n, len(ds))):
            r = ds[i]
            add(i, _pair(r["sentence1"], r["sentence2"]), [Question(f"paws-{split}-{i}", "noul", "Is Text B a paraphrase of Text A (same meaning, not just shared words)?", list(NOUL_OPTIONS), int(r["label"] == 1))])
    elif name == "cola":
        ds = load_dataset("nyu-mll/glue", "cola", split=split)
        for i in rng.sample(range(len(ds)), min(n, len(ds))):
            r = ds[i]
            add(i, r["sentence"], [Question(f"cola-{split}-{i}", "noul", "Is this sentence grammatically acceptable English?", list(NOUL_OPTIONS), int(r["label"] == 1))])
    elif name == "emotion":
        ds = load_dataset("dair-ai/emotion", "split", split=split)
        for i in rng.sample(range(len(ds)), min(n, len(ds))):
            r = ds[i]; lab = int(r["label"])
            add(i, r["text"], [Question(f"emotion-{split}-{i}-which", "choice", "Which emotion does the writer express?", EMOTIONS, lab),
                                Question(f"emotion-{split}-{i}-neg", "noul", "Is the writer expressing a negative emotion (sadness, anger or fear)?", list(NOUL_OPTIONS), int(lab in (0, 3, 4)))])
    elif name == "clinc":
        ds = load_dataset("clinc/clinc_oos", "plus", split=split)
        names = [x.replace("_", " ") for x in ds.features["intent"].names]
        for i in rng.sample(range(len(ds)), min(n, len(ds))):
            r = ds[i]; lab = int(r["intent"])
            # 151 intents do not fit a 512-token window beside the state; 20-way subset (gold + 19 sampled)
            others = rng.sample([j for j in range(len(names)) if j != lab], 19)
            opts = sorted(others + [lab])
            add(i, r["text"], [Question(f"clinc-{split}-{i}", "choice", "Which assistant intent does the user utterance express?", [names[j] for j in opts], opts.index(lab)),
                                Question(f"clinc-{split}-{i}-oos", "noul", "Is this utterance outside the assistant's supported scope?", list(NOUL_OPTIONS), int(names[lab] == "oos"))])
    elif name == "ag_news":
        ds = load_dataset("fancyzhx/ag_news", split=split)
        for i in rng.sample(range(len(ds)), min(n, len(ds))):
            r = ds[i]
            add(i, r["text"], [Question(f"ag-{split}-{i}", "choice", "Which section of a newspaper does this article belong to?", AG, int(r["label"]))])
    elif name == "dbpedia":
        ds = load_dataset("fancyzhx/dbpedia_14", split=split)
        for i in rng.sample(range(len(ds)), min(n, len(ds))):
            r = ds[i]
            add(i, f"{r['title']}: {r['content']}", [Question(f"dbp-{split}-{i}", "choice", "What kind of entity does this encyclopedia entry describe?", DBP, int(r["label"]))])
    elif name == "yelp":
        ds = load_dataset("Yelp/yelp_review_full", split=split)
        for i in rng.sample(range(len(ds)), min(n, len(ds))):
            r = ds[i]; lab = int(r["label"])
            add(i, r["text"], [Question(f"yelp-{split}-{i}-stars", "score", "How many stars did the reviewer give this business?", STARS, lab),
                                Question(f"yelp-{split}-{i}-rec", "noul", "Would the reviewer recommend this business?", list(NOUL_OPTIONS), int(lab >= 3))])
    elif name == "imdb":
        ds = load_dataset("stanfordnlp/imdb", split=split)
        for i in rng.sample(range(len(ds)), min(n, len(ds))):
            r = ds[i]
            add(i, r["text"][:1500], [Question(f"imdb-{split}-{i}", "noul", "Did the reviewer like the film?", list(NOUL_OPTIONS), int(r["label"] == 1))])
    elif name == "counterfactual":
        ds = load_dataset("SetFit/amazon_counterfactual_en", split=split)
        for i in rng.sample(range(len(ds)), min(n, len(ds))):
            r = ds[i]
            add(i, r["text"], [Question(f"cf-{split}-{i}", "noul", "Does the sentence describe something that did NOT happen (a counterfactual, e.g. 'I wish it had…')?", list(NOUL_OPTIONS), int(r["label"] == 1))])
    else:
        raise ValueError(name)
    return out


SOURCES = {"mnli": ("train", "validation_matched"), "rte": ("train", "validation"), "qnli": ("train", "validation"), "mrpc": ("train", "validation"), "paws": ("train", "test"),
           "cola": ("train", "validation"), "emotion": ("train", "test"), "clinc": ("train", "test"), "ag_news": ("train", "test"), "dbpedia": ("train", "test"),
           "yelp": ("train", "test"), "imdb": ("train", "test"), "counterfactual": ("train", "test")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data")
    ap.add_argument("--out", default="data-multi")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    train = read_jsonl(os.path.join(a.base, "train.jsonl"))
    val = read_jsonl(os.path.join(a.base, "val.jsonl"))
    test = read_jsonl(os.path.join(a.base, "test.jsonl"))
    counts = {"base": {"train": len(train), "test": len(test)}}
    for src in a.sources.split(","):
        tr_split, te_split = SOURCES[src]
        tr = build_source(src, tr_split, a.n_train + 100, rng)
        te = build_source(src, te_split, a.n_test, rng)
        val += tr[:100]
        train += tr[100:]
        test += te
        counts[src] = {"train": len(tr) - 100, "test": len(te), "questions": sum(len(e.questions) for e in tr)}
        print(src, counts[src], flush=True)
    rng.shuffle(train)
    rng.shuffle(test)
    write_jsonl(os.path.join(a.out, "train.jsonl"), train)
    write_jsonl(os.path.join(a.out, "val.jsonl"), val)
    write_jsonl(os.path.join(a.out, "test.jsonl"), test)
    shutil.copy(os.path.join(a.base, "ood-test.jsonl"), os.path.join(a.out, "ood-test.jsonl"))  # unchanged, for comparability
    counts["total"] = {"train_states": len(train), "train_questions": sum(len(e.questions) for e in train), "val": len(val), "test_states": len(test), "families": len(a.sources.split(",")) + 3}
    json.dump(counts, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(counts["total"]))


if __name__ == "__main__":
    main()
