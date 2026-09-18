import os
import random

import pytest

from typed_decisions.schema import Example, Question, NOUL_OPTIONS, write_jsonl

COLORS = ["red", "green", "blue", "yellow"]
SIZES = ["tiny", "small", "medium", "large", "huge"]


def synth_example(rng: random.Random, i: int, split: str) -> Example:
    c = rng.randrange(len(COLORS))
    s = rng.randrange(len(SIZES))
    state = f"item {i}: a {SIZES[s]} {COLORS[c]} box"
    return Example(state=state, source="synth", meta={"i": i}, questions=[
        Question(f"{split}-{i}-color", "choice", "What colour is the box?", COLORS, c),
        Question(f"{split}-{i}-size", "score", "How big is the box?", SIZES, s),
        Question(f"{split}-{i}-red", "noul", "The box is red.", list(NOUL_OPTIONS), int(c == 0)),
    ])


@pytest.fixture(scope="session")
def synth_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    rng = random.Random(0)
    write_jsonl(d / "train.jsonl", [synth_example(rng, i, "train") for i in range(400)])
    write_jsonl(d / "val.jsonl", [synth_example(rng, i, "val") for i in range(40)])
    write_jsonl(d / "test.jsonl", [synth_example(rng, i, "test") for i in range(60)])
    return str(d)
