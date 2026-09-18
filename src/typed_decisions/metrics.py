"""Accuracy, Brier, ECE and score-MAE over predicted distributions. Every number comes with its
`n`; a group with n = 0 is reported as `null`, never as 0.0 or 1.0 (CLAUDE.md 8 問 #1)."""

from __future__ import annotations

import math
from collections import defaultdict


def ece(confs: list[float], correct: list[bool], bins: int = 15) -> float | None:
    if not confs:
        return None
    tot = [0] * bins
    acc = [0.0] * bins
    con = [0.0] * bins
    for c, ok in zip(confs, correct):
        b = min(bins - 1, int(c * bins))
        tot[b] += 1
        acc[b] += float(ok)
        con[b] += c
    n = len(confs)
    return sum(abs(acc[b] / tot[b] - con[b] / tot[b]) * tot[b] / n for b in range(bins) if tot[b])


def summarize(records: list[dict]) -> dict:
    """records: {kind, source, probs, gold}. Returns per-kind, per-source and overall metrics."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups["all"].append(r)
        groups[f"kind={r['kind']}"].append(r)
        groups[f"source={r['source']}"].append(r)
        groups[f"{r['source']}/{r['kind']}"].append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        n = len(rs)
        if n == 0:
            out[g] = {"n": 0}
            continue
        pred = [max(range(len(r["probs"])), key=lambda i: r["probs"][i]) for r in rs]
        correct = [p == r["gold"] for p, r in zip(pred, rs)]
        conf = [max(r["probs"]) for r in rs]
        brier = sum(sum((p - (1.0 if i == r["gold"] else 0.0)) ** 2 for i, p in enumerate(r["probs"])) for r in rs) / n
        nll = sum(-math.log(max(r["probs"][r["gold"]], 1e-12)) for r in rs) / n
        m = {"n": n, "acc": sum(correct) / n, "brier": brier, "nll": nll, "ece": ece(conf, correct), "mean_conf": sum(conf) / n}
        scores = [r for r in rs if r["kind"] == "score"]
        if scores and all(r["kind"] == "score" for r in rs):
            m["score_mae"] = sum(abs(sum(i * p for i, p in enumerate(r["probs"])) - r["gold"]) for r in rs) / n
        if all(r["kind"] == "noul" for r in rs):
            # noul has no separate confidence: its calibration is the Brier of p(yes) against the label
            m["noul_brier"] = sum((r["probs"][1] - r["gold"]) ** 2 for r in rs) / n
        out[g] = m
    return out


def fit_temperature(logit_groups: list[list[float]], golds: list[int], grid=None) -> float:
    """Single scalar temperature minimising NLL on held-out logits (Guo et al. 2017), grid search
    so it is deterministic and dependency-free."""
    import numpy as np
    grid = grid if grid is not None else [x / 20 for x in range(4, 80)]  # 0.2 .. 3.95
    best, best_nll = 1.0, float("inf")
    for t in grid:
        nll = 0.0
        for lg, g in zip(logit_groups, golds):
            z = np.asarray(lg, dtype=np.float64) / t
            z = z - z.max()
            nll += -(z[g] - math.log(np.exp(z).sum()))
        if nll < best_nll:
            best, best_nll = t, nll
    return best
