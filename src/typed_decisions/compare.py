"""Render the side-by-side tables in the README from report.json files.

    python -m typed_decisions.compare reports/enc-*.json reports/dllm-*.json
"""

from __future__ import annotations

import json
import sys


def _m(rep: dict, key: str):
    return rep.get(key) or {}


def row_accuracy(name: str, rep: dict, key: str) -> str:
    m = _m(rep, key)
    if not m:
        return f"| {name} | — | | | | | | | |"
    a = m["all"]
    g = lambda k, f="acc": (f"{m[k][f]:.3f}" if k in m and f in m[k] else "—")
    return (f"| {name} | {a['n']} | **{a['acc']:.3f}** | {a['brier']:.3f} | {a['ece']:.3f} | {g('banking77/choice')} | {g('banking77/noul')} | "
            f"{g('sst5/score')} / {g('sst5/score', 'score_mae')} | {g('sst5/choice')} | {g('boolq/noul')} |")


def row_train(name: str, rep: dict) -> str:
    t = rep["train"]
    d = rep["data"]
    return (f"| {name} | {d['train_states']} / {d['train_questions']} | {t['steps']} | {t['wall_s']:.0f} s | {t['seq_tokens_per_s']:.0f} | "
            f"{t['peak_mem_gib']:.1f} | ${t['usd_h100']:.3f} | ${t.get('usd_per_1k_train_questions', 0):.4f} | {t['loss_first']:.2f} → {t['loss_last10_mean']:.2f} |")


def row_latency(name: str, rep: dict) -> str:
    cells = []
    for l in rep["latency"]:
        if "error" in l:
            cells.append(f"{l['timed']} N={l['n_questions']}: does not fit")
            continue
        tag = f"{l['timed']} N={l['n_questions']}" + (f" steps={l['steps']}" if l.get("steps", 1) != 1 else "") + (" (77-opt pool)" if l["max_options"] is None else "")
        cells.append(f"{tag}: {l['p50_ms']:.0f} / {l['p95_ms']:.0f} ms, {l['decisions_per_s_at_p50']:.0f} dec/s")
    tp = ", ".join(f"b{x['batch']} {x['questions_per_s']:.0f} q/s" for x in rep["throughput"])
    return f"| {name} | " + "<br>".join(cells) + f" | {tp} |"


def main(argv):
    reps = [(p.split("/")[-1].rsplit("-2026", 1)[0], json.load(open(p))) for p in argv]
    print("### accuracy / calibration (test, 1,500 states)\n")
    print("| run | n questions | acc | Brier | ECE | b77 intent (77) | b77 card noul | sst5 level acc / MAE | sst5 polarity | boolq noul |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, r in reps:
        keys = [k for k in r if k.startswith("metrics_") and k != "metrics_train_subset"]
        for k in keys:
            print(row_accuracy(f"{name} · {k[8:]}", r, k))
    print("\n### training cost (H100 80GB, Modal, $0.001097/s)\n")
    print("| run | train states / questions | steps | wall | seq-tok/s | peak GiB | USD | USD per 1k train questions | loss first → last-10 mean |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, r in reps:
        print(row_train(name, r))
    print("\n### latency (batch 1, one state, N questions packed; p50 / p95) and throughput\n")
    print("| run | latency rows | throughput (states with their own 1–3 questions) |")
    print("|---|---|---|")
    for name, r in reps:
        print(row_latency(name, r))


if __name__ == "__main__":
    main(sys.argv[1:])
