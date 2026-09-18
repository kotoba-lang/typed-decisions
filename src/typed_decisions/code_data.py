"""Code decisions from the workspace symbol index (`.kotoba-cache/symbol-index.tsv`, v5).

The unison-like reading of a program: a definition is a node with a content hash, and wiring it
means choosing which existing definitions it references. That choice is a `Choice` over candidate
definitions — an option is a definition (name · namespace · docstring · hash), the answer is a
pointer, nothing is generated. This loader turns the index into that task:

  state     = "<ns>/<name>  doc  — references so far: r1, r2, ..."   (one reference held out)
  choice    = which of these definitions does it ALSO reference?     options = held-out ref + k distractors
              distractors: definitions referenced by *sibling* defs in the same namespace but not by this one
              (the same neighbourhood a type-directed candidate list would come from; no types in the index)
  noul      = "does it reference <x>?"  for the held-out ref (yes) and one distractor (no)

Split by namespace hash so a test namespace is never seen in training (a defs-in-ns leak would
otherwise make the distractor pool give the answer away). Only `orgs/kotoba-lang` rows with >= 2
references. The docstring column is often empty; the state then carries only the name and the
references — the point is what a model can decide from the index alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict

from .schema import Example, Question, NOUL_OPTIONS, write_jsonl

INDEX = ".kotoba-cache/symbol-index.tsv"
KINDS = {"defn", "defn-", "def", "defmethod", "defonce"}


def load_defs(top: str, prefix: str = "orgs/") -> tuple[list[dict], dict[str, dict]]:
    defs, by_fq = [], {}
    with open(os.path.join(top, INDEX)) as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 8 or c[1] not in KINDS or not c[2].startswith(prefix):
                continue
            sym, kind, path, ln, ns, doc, h, refs = c[:8]
            raw = [r for r in refs.split(",") if r and not r.endswith("*")]  # `MAP*` / `SET*` are shape tags, not references
            d = {"sym": sym, "kind": kind, "path": path, "line": int(ln or 0), "ns": ns, "doc": doc.strip(), "hash": h, "raw_refs": raw, "fq": f"{ns}/{sym}"}
            defs.append(d)
            by_fq[d["fq"]] = d
    # resolve each raw reference to a known definition: `a.b/c` as-is, bare `c` to the same namespace
    for d in defs:
        seen, out = set(), []
        for r in d["raw_refs"]:
            fq = r if "/" in r else f"{d['ns']}/{r}"
            if fq in by_fq and fq != d["fq"] and fq not in seen:
                seen.add(fq)
                out.append(fq)
        d["refs"] = out
    return defs, by_fq


def describe(fq: str, by_fq: dict[str, dict]) -> str:
    d = by_fq.get(fq)
    if d and d["doc"]:
        return f"{fq} — {d['doc'][:60]}"
    return fq


def build(top: str, out_dir: str, n_options: int = 6, seed: int = 0, test_frac: float = 0.1, max_per_ns: int = 40) -> dict:
    rng = random.Random(seed)
    defs, by_fq = load_defs(top)
    by_ns: dict[str, list[dict]] = defaultdict(list)
    for d in defs:
        by_ns[d["ns"]].append(d)
    train, test = [], []
    skipped = {"lt2refs": 0, "no_distractors": 0}
    for ns, ds in sorted(by_ns.items()):
        ns_refs = set(r for d in ds for r in d["refs"])
        is_test = int(hashlib.sha1(ns.encode()).hexdigest()[:8], 16) % 1000 < test_frac * 1000
        picked = ds if len(ds) <= max_per_ns else rng.sample(ds, max_per_ns)
        for d in picked:
            if len(d["refs"]) < 2:
                skipped["lt2refs"] += 1
                continue
            held = rng.choice(d["refs"])
            others = [r for r in d["refs"] if r != held]
            pool = sorted(ns_refs - set(d["refs"]))
            if len(pool) < 2:
                skipped["no_distractors"] += 1
                continue
            distractors = rng.sample(pool, min(n_options - 1, len(pool)))
            opts = [held] + distractors
            rng.shuffle(opts)
            state = f"definition {d['fq']}" + (f"\n{d['doc'][:300]}" if d["doc"] else "") + "\nreferences so far: " + ", ".join(others[:12])
            neg = rng.choice(distractors)
            meta = {"ns": ns, "fq": d["fq"], "hash": d["hash"]}
            # the choice and the nouls go on SEPARATE examples: packed together, the `yes` noul names the
            # held-out reference and the choice becomes string matching (measured: train loss 0.0000)
            bucket = test if is_test else train
            bucket.append(Example(state=state, source="code", meta=meta, questions=[
                Question(f"code-{d['hash']}-ref", "choice", "Which of these definitions does it also reference?", [describe(o, by_fq) for o in opts], opts.index(held))]))
            pair = [Question(f"code-{d['hash']}-yes", "noul", f"Does it reference {held}?", list(NOUL_OPTIONS), 1),
                    Question(f"code-{d['hash']}-no", "noul", f"Does it reference {neg}?", list(NOUL_OPTIONS), 0)]
            rng.shuffle(pair)
            bucket.append(Example(state=state, source="code-noul", meta=meta, questions=pair))
    rng.shuffle(train)
    rng.shuffle(test)
    os.makedirs(out_dir, exist_ok=True)
    val = train[:400]
    train = train[400:]
    write_jsonl(os.path.join(out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(out_dir, "val.jsonl"), val)
    write_jsonl(os.path.join(out_dir, "test.jsonl"), test)
    return {"defs": len(defs), "namespaces": len(by_ns), "train": len(train), "val": len(val), "test": len(test), "skipped": skipped,
            "test_namespaces": len(set(e.meta["ns"] for e in test)), "resolved_refs_mean": sum(len(d["refs"]) for d in defs) / max(1, len(defs)), "options_mean": sum(len(e.questions[0].options) for e in test if e.source == "code") / max(1, sum(1 for e in test if e.source == "code"))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", default="../../..")
    ap.add_argument("--out", default="data-code")
    ap.add_argument("--n-options", type=int, default=6)
    ap.add_argument("--max-per-ns", type=int, default=40)
    a = ap.parse_args(argv)
    print(json.dumps(build(a.top, a.out, a.n_options, max_per_ns=a.max_per_ns), indent=1))


if __name__ == "__main__":
    main()
