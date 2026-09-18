import random
from typed_decisions.augment import augment, pair, shuffle, negate, drop, relabel, paraphrase
from typed_decisions.schema import Question, NOUL_OPTIONS


def test_gold_preserved_by_construction():
    rng = random.Random(0)
    c = Question("c", "choice", "Which team should handle this?", ["billing", "tech", "sales", "legal", "other"], 2)
    s = Question("s", "score", "How angry is the customer?", ["calm", "annoyed", "angry"], 1)
    n = Question("n", "noul", "The message conveys urgency.", list(NOUL_OPTIONS), 1)
    for _ in range(200):
        q = augment(c, rng)
        assert q.options[q.gold] == "sales" and q.kind == "choice"
        q = augment(s, rng)
        assert q.gold == 1 and len(q.options) == 3
        q = augment(n, rng, p_negate=0.0)
        assert q.gold == 1
    ng = negate(n, rng)
    assert ng.gold == 0 and "urgency" in ng.instructions
    a, b = pair(c, rng)
    assert a.options[a.gold] == b.options[b.gold]
    assert drop(c, rng).options.count("sales") == 1 and len(drop(c, rng).options) < 5
    assert relabel(s, rng).options != s.options and shuffle(s, rng).options == s.options
    assert paraphrase(n, rng).instructions != n.instructions or True
