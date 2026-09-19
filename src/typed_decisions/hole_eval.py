"""Ask live Jev the held-out `code-holes` questions and score them against the commit's gold.

    python -m typed_decisions.hole_eval --data data-holes/test.jsonl --limit 200 --out reports/hole-eval-jev.json

Per-kind accuracy, gold rank, and the confidence split — the same reading as 第6/7反復, on holes
from repositories that never appear in the 24 verified pairs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

from .jev_holes import jev_choice, openrouter_key, ranked
from .schema import read_jsonl


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data-holes/test.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=os.environ.get("JEV_MODEL", "typesafe/jev-1.13"))
    ap.add_argument("--max-state", type=int, default=24000, help="chars; Jev context is 32K tokens")
    ap.add_argument("--out", default="reports/hole-eval-jev.json")
    a = ap.parse_args(argv)
    exs = read_jsonl(a.data)
    random.Random(a.seed).shuffle(exs)
    exs = exs[: a.limit]
    key = openrouter_key()
    rows, cost, t0 = [], 0.0, time.time()
    for e in exs:
        q = e.questions[0]
        state = e.state if len(e.state) <= a.max_state else e.state[: a.max_state] + "\n[truncated]"
        row = {"repo": e.meta["repo"], "sha": e.meta["sha"], "kind": e.meta["kind"], "old": e.meta["old"], "gold": q.options[q.gold],
               "n_options": len(q.options), "has_test": e.meta["has_test"], "verified": e.meta["verified"]}
        try:
            ans = jev_choice(state, q.instructions, q.options, a.model, key)
            order = ranked(ans["probabilities"])
            row.update({"top1": ans["choice"], "correct": ans["choice"] == row["gold"],
                        "gold_rank": order.index(row["gold"]) + 1 if row["gold"] in order else None,
                        "confidence": ans["confidence"], "p_gold": ans["probabilities"].get(row["gold"]), "cost": ans["cost"]})
            cost += ans["cost"] or 0.0
        except Exception as ex:  # noqa: BLE001
            row["error"] = str(ex)[:200]
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("repo", "kind", "old", "gold", "top1", "correct", "gold_rank", "confidence", "error")}), flush=True)

    ok = [r for r in rows if "correct" in r]
    def acc(rs):
        return round(sum(r["correct"] for r in rs) / len(rs), 3) if rs else None
    by_kind = {k: {"n": len(v), "top1": acc(v), "top3": round(sum(1 for r in v if r["gold_rank"] and r["gold_rank"] <= 3) / len(v), 3),
                   "mean_chance": round(sum(1 / r["n_options"] for r in v) / len(v), 4)}
               for k, v in {k: [r for r in ok if r["kind"] == k] for k in sorted({r["kind"] for r in ok})}.items()}
    c = [r["confidence"] for r in ok if r["correct"]]
    w = [r["confidence"] for r in ok if not r["correct"]]
    gates = {}
    for thr in (0.5, 0.6, 0.7, 0.8):
        auto = [r for r in ok if (r["confidence"] or 0) >= thr]
        gates[str(thr)] = {"auto": len(auto), "auto_correct": sum(r["correct"] for r in auto), "auto_wrong": sum(not r["correct"] for r in auto), "escalated": len(ok) - len(auto)}
    summary = {"n": len(ok), "errors": len(rows) - len(ok), "top1": acc(ok), "by_kind": by_kind,
               "by_verified": {v: {"n": len(rs), "top1": acc(rs)} for v, rs in {v: [r for r in ok if r["verified"] == v] for v in ("commit", "test")}.items() if rs},
               "with_test": {"n": sum(r["has_test"] for r in ok), "top1": acc([r for r in ok if r["has_test"]])},
               "without_test": {"n": sum(not r["has_test"] for r in ok), "top1": acc([r for r in ok if not r["has_test"]])},
               "conf_correct_mean": round(sum(c) / len(c), 3) if c else None, "conf_wrong_mean": round(sum(w) / len(w), 3) if w else None,
               "conf_correct_min": min(c) if c else None, "conf_wrong_max": max(w) if w else None, "gates": gates,
               "api_cost_usd": round(cost, 4), "wall_s": round(time.time() - t0, 1), "model": a.model}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
