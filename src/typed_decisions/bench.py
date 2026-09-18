"""Latency / throughput measurement shared by both backbones.

`latency(fn, pool, n_questions)`: one state, N questions packed on it, batch 1 — Jev's unit of
work ("many decisions on one program state in a single pass"). Reports p50/p95 wall ms over
`repeats` after `warmup`, and decisions per second at that N.

`throughput(fn, examples, batch)`: many states with their own questions, batched — the serving
number. Both synchronise the device before reading the clock.
"""

from __future__ import annotations

import random
import statistics
import time

import torch

from .schema import Example, Question


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elif str(device).startswith("mps"):
        torch.mps.synchronize()


def pack(pool: list[Example], n_questions: int, rng: random.Random, state: str | None = None, max_options: int | None = None) -> tuple[str, list[Question]]:
    qs = [q for e in pool for q in e.questions if max_options is None or len(q.options) <= max_options]
    st = state if state is not None else rng.choice(pool).state
    picked = [rng.choice(qs) for _ in range(n_questions)]
    return st, [Question(f"pack-{i}", q.kind, q.instructions, q.options, q.gold) for i, q in enumerate(picked)]


def latency(fn, pool: list[Example], n_questions: int, device, repeats: int = 20, warmup: int = 3, seed: int = 0, max_options: int | None = None, prep=None) -> dict:
    """max_options restricts the question pool (banking77's 77-option intent question is ~350
    tokens by itself; 100 of them do not fit an 8k encoder, and are not a typical workflow question)."""
    rng = random.Random(seed)
    items = [pack(pool, n_questions, rng, max_options=max_options) for _ in range(repeats + warmup)]
    tokens_hint = sum(len(q.options) for _, qs in items[warmup:] for q in qs) / repeats
    # prep (tokenise + collate) outside the clock when given: that row is the model's forward alone;
    # without prep the row is end-to-end from text, which is what a caller pays
    prepped = [prep([it]) if prep else [it] for it in items]
    for it in prepped[:warmup]:
        fn(it)
        _sync(device)
    times = []
    for it in prepped[warmup:]:
        _sync(device)
        t0 = time.perf_counter()
        fn(it)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p50 = statistics.median(times)
    p95 = times[min(len(times) - 1, int(round(0.95 * (len(times) - 1))))]
    return {"n_questions": n_questions, "timed": "forward" if prep else "e2e", "max_options": max_options, "mean_options_per_pack": tokens_hint, "repeats": repeats,
            "p50_ms": p50, "p95_ms": p95, "min_ms": times[0], "decisions_per_s_at_p50": n_questions / (p50 / 1000)}


def throughput(fn, examples: list[Example], batch: int, device, max_examples: int = 256) -> dict:
    ex = examples[:max_examples]
    fn([(e.state, e.questions) for e in ex[:batch]])
    _sync(device)
    t0 = time.perf_counter()
    nq = 0
    for i in range(0, len(ex), batch):
        chunk = ex[i : i + batch]
        fn([(e.state, e.questions) for e in chunk])
        nq += sum(len(e.questions) for e in chunk)
    _sync(device)
    dt = time.perf_counter() - t0
    return {"batch": batch, "states": len(ex), "questions": nq, "wall_s": dt, "questions_per_s": nq / dt, "states_per_s": len(ex) / dt}
