"""Label the corpus with a chat-API teacher (default: kotoba.cloud qwen3.8-flash-next-whitehacker).

Measured 2026-09-18 on that lane: no `logprobs`, `n` must be 1, 8 samples at temperature 1.0 were
identical — so the teacher yields a HARD label per question, not a distribution. This module
therefore writes {qid: letter-index} plus the teacher's optional self-reported confidence, and the
student distils from one-hot teacher labels (KL to a one-hot is CE). Nothing here is a probability
from the teacher.

    python -m typed_decisions.teacher --data data --split test --limit 300 --out data/teacher-test.jsonl
    python -m typed_decisions.teacher --data data --split train --out data/teacher-train.jsonl

Env: KOTOBA_API_TOKEN (or --env-file ~/.hermes/.env). Requests are cached by qid in --out so a rerun
resumes. Prints agreement with gold per source/kind, throughput, and token usage.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from .schema import Example, read_jsonl
from .metrics import summarize

LETTERS = [chr(ord("A") + i) for i in range(26)] + [f"A{chr(ord('A') + i)}" for i in range(26)] + [f"B{chr(ord('A') + i)}" for i in range(26)] + [str(i) for i in range(100, 300)]


def prompt_for(e: Example) -> str:
    lines = ["<state>", e.state, "</state>", "Answer each question with the label of one option, one answer per line as `A1: X`. No explanation."]
    for qi, q in enumerate(e.questions):
        opts = "  ".join(f"{LETTERS[j]}) {o}" for j, o in enumerate(q.options))
        lines.append(f"Q{qi + 1}. {q.instructions}\nOptions: {opts}")
    return "\n".join(lines)


def parse(text: str, e: Example) -> list[int | None]:
    """Line-oriented: the i-th answer line (`A1: X`, `Q1. X`, or a bare label) is the i-th question's
    answer — the teacher often numbers every line `A1:` (measured: 30/300 states). A line that is
    prose (explanation instead of a label) yields None for that question."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    labels = []
    for l in lines:
        m = re.match(r"^(?:[AQ]\d+\s*[:：.)]\s*)?([A-Z]{1,2}|\d{3})\)?\s*$", l)
        labels.append(m.group(1) if m else None)
    out = []
    for qi, q in enumerate(e.questions):
        lab = labels[qi] if qi < len(labels) else None
        idx = LETTERS.index(lab) if lab in LETTERS else None
        out.append(idx if idx is not None and idx < len(q.options) else None)
    return out


def call(base_url: str, token: str, model: str, prompt: str, timeout: float = 120) -> tuple[str, dict, float]:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 64, "temperature": 0}).encode()
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                                          # kotoba.cloud sits behind Cloudflare, whose browser-integrity check answers 403 (error 1010) to
                                          # urllib's default "Python-urllib/3.x" while curl's default passes (measured 2026-09-18). An honest
                                          # client name is what an API client should send anyway.
                                          "User-Agent": "typed-decisions/0.0.1 (+https://github.com/kotoba-lang/typed-decisions)"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    if "error" in d:
        raise RuntimeError(json.dumps(d["error"])[:300])
    return d["choices"][0]["message"]["content"], d.get("usage", {}), time.perf_counter() - t0


def load_token(env_file: str | None) -> str:
    t = os.environ.get("KOTOBA_API_TOKEN")
    if not t and env_file:
        for line in open(os.path.expanduser(env_file)):
            if line.startswith("KOTOBA_API_TOKEN="):
                t = line.split("=", 1)[1].strip().strip('"')
    if not t:
        raise SystemExit("REFUSE: KOTOBA_API_TOKEN not set")
    return t


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sources", default="", help="comma-separated source filter, e.g. sst5,boolq")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3.8-flash-next-whitehacker")
    ap.add_argument("--base-url", default="https://api.kotoba.cloud/v1")
    ap.add_argument("--env-file", default="~/.hermes/.env")
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args(argv)
    token = load_token(a.env_file)
    ex = read_jsonl(os.path.join(a.data, f"{a.split}.jsonl"))
    if a.sources:
        keep = set(a.sources.split(","))
        ex = [e for e in ex if e.source in keep]
    if a.limit:
        ex = ex[: a.limit]
    done: dict[str, dict] = {}
    if os.path.exists(a.out):
        for l in open(a.out):
            d = json.loads(l)
            done[d["state_key"]] = d
    todo = [e for e in ex if e.questions[0].qid not in done]
    print(f"{len(ex)} states, {len(done)} cached, {len(todo)} to label", flush=True)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    lat, errors = [], 0
    t0 = time.perf_counter()
    with open(a.out, "a") as f, ThreadPoolExecutor(a.concurrency) as pool:
        futs = {pool.submit(call, a.base_url, token, a.model, prompt_for(e)): e for e in todo}
        for i, fu in enumerate(as_completed(futs)):
            e = futs[fu]
            try:
                text, u, dt = fu.result()
            except Exception as err:
                errors += 1
                print("ERR", e.questions[0].qid, str(err)[:120], flush=True)
                continue
            lat.append(dt)
            for k in usage:
                usage[k] += int(u.get(k, 0))
            rec = {"state_key": e.questions[0].qid, "labels": {q.qid: v for q, v in zip(e.questions, parse(text, e))}, "raw": text[:200]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[rec["state_key"]] = rec
            if i % 25 == 0:
                print(f"{i}/{len(todo)} lat p50 {statistics.median(lat):.1f}s", flush=True)
    wall = time.perf_counter() - t0
    recs = []
    unparsed = 0
    for e in ex:
        d = done.get(e.questions[0].qid)
        if not d:
            continue
        labels = {q.qid: v for q, v in zip(e.questions, parse(d["raw"], e))} if d.get("raw") else d["labels"]  # re-parse: parser fixes apply to cached rows
        for q in e.questions:
            v = labels.get(q.qid)
            if v is None:
                unparsed += 1
                continue
            probs = [0.0] * len(q.options)
            probs[v] = 1.0
            recs.append({"kind": q.kind, "source": e.source, "probs": probs, "gold": q.gold})
    rep = {"model": a.model, "states": len(ex), "labelled_now": len(todo) - errors, "errors": errors, "unparsed_questions": unparsed,
           "wall_s": wall, "req_per_s": (len(todo) - errors) / wall if wall > 0 and todo else None,
           "latency_p50_s": statistics.median(lat) if lat else None, "latency_p95_s": sorted(lat)[int(0.95 * (len(lat) - 1))] if lat else None,
           "usage": usage, "agreement_with_gold": {k: {kk: v[kk] for kk in ("n", "acc")} for k, v in summarize(recs).items()}}
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    main()
