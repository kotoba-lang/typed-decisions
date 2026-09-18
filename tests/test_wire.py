from typed_decisions.schema import Question
from typed_decisions.wire import proposal, to_edn, memo_key


def test_proposal_is_a_pointer_not_text():
    q = Question("q", "choice", "Which of these definitions does it also reference?", ["a.b/x — doc", "a.b/y", "a.b/z"], 1)
    hashes = ["h1", "h2", "h3"]
    p = proposal("a.b/f", "definition a.b/f", q, [0.1, 0.8, 0.1], hashes, admit_noul=0.91)
    assert p["reference"] == {"fq": "a.b/y", "hash": "h2"}
    assert p["admit?"]["decision"] == "autonomous"
    assert proposal("a.b/f", "s", q, [0.1, 0.8, 0.1], hashes, admit_noul=0.5)["admit?"]["decision"] == "escalate"
    e = to_edn(p)
    assert e.startswith("{:proposal/kind \"wire-reference\"") and ":memo-key" in e
    # memo key is a function of content only: same inputs -> same key, any option hash change -> different key
    assert memo_key("s", q, hashes) == memo_key("s", q, list(hashes))
    assert memo_key("s", q, hashes) != memo_key("s", q, ["h1", "h2", "h9"])


def test_kaizen_issue_id_is_the_memo_key():
    from typed_decisions.wire import to_kaizen_issue
    q = Question("q", "choice", "Which?", ["a.b/x", "a.b/y"], 1)
    p = proposal("a.b/f", "s", q, [0.2, 0.8], ["h1", "h2"], admit_noul=0.9)
    i = to_kaizen_issue(p, "kotoba-lang", "typed-decisions")
    assert i["id"] == "kaizen:typed-decisions:" + p["memo-key"][:16] + ":once" and i["kind"] == "typed-decision" and "nothing was executed" in i["body"]
