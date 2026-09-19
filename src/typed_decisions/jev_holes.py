"""Can a choice-only model (TypeSafe Jev) drive a kotoba refactor when the driver loop, the candidate
set and the verifier are supplied from outside?

The test bed is `data/kotoba-tasks.json` (mine_kotoba_tasks.py): real fix commits, each verified
FAIL(before)/PASS(after) under kbb. This script does NOT ask the model to write code. It

  1. diffs src_before / src_after at token level and keeps only *substitution holes* — a hunk that
     deletes one code-like token and inserts one (keywords, symbols, namespaced names). Prose
     hunks (docstrings) and insertions are counted but not attempted: they are the part a choice
     model cannot do, and the count is the finding.
  2. builds the option list for each hole from tokens of the same kind that occur in the broken
     module or in the test (the test is the spec and names what the module must provide), minus
     the token being replaced (a hole means "this changes"; the first run left it in and the model
     picked "no change" with high confidence in every wrong non-trivial case). Nothing is looked
     up elsewhere, and the gold token is NOT added if it is absent — an unreachable gold is
     recorded as such, not smuggled in.
  3. asks Jev one `choice` per hole (state = commit intent + test + module with `<<HOLE>>` + the
     failing output), independently.
  4. applies the top-1 answer to every hole and runs the real test under kbb. If it fails,
     backtracks: lowest-confidence hole first, next option from Jev's own distribution, re-run,
     within a fixed budget of runs.

Every number reported comes from a live API call or a real kbb run; chance is 1/|options|.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import time
import urllib.request

from .mine_kotoba_tasks import run_case

TOK = re.compile(r'"(?:[^"\\]|\\.)*"|;[^\n]*|[^\s()\[\]{}"]+|[()\[\]{}]')
CODEISH = re.compile(r"^(?::[\w.\-!?*+<>=/]+|[A-Za-z_*+!?<>=][\w.\-!?*+<>=/']*)$")
ENGLISH_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "that", "this", "for", "on", "with", "as", "by", "at", "from", "not", "be", "are", "was", "were", "has", "have", "had", "its", "if", "so", "we", "one", "into", "when", "which", "than", "then", "out", "up", "no", "yes"}


def tokens(src: str) -> list[str]:
    return TOK.findall(src)


def kind_of(tok: str) -> str | None:
    if tok.startswith(";"):
        return None
    if tok.startswith('"'):
        # a whitespace-free string literal is a name (wire name, key, id), not prose
        return "str" if len(tok) > 2 and not re.search(r"\s", tok) else None
    if not CODEISH.match(tok):
        return None
    if tok.startswith(":"):
        return "keyword"
    if tok.lower() in ENGLISH_STOP and "-" not in tok:
        return None
    return "symbol"


def holes_of(before: str, after: str) -> tuple[list[dict], dict]:
    ta, tb = tokens(before), tokens(after)
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    holes, shape = [], {"replace1": 0, "replace_n": 0, "insert": 0, "delete": 0, "ins_tokens": 0, "del_tokens": 0}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        shape["ins_tokens"] += j2 - j1
        shape["del_tokens"] += i2 - i1
        if op == "replace" and i2 - i1 == 1 and j2 - j1 == 1:
            k = kind_of(ta[i1])
            if k and kind_of(tb[j1]) == k:
                shape["replace1"] += 1
                holes.append({"pos": i1, "old": ta[i1], "gold": tb[j1], "kind": k})
                continue
        shape[op if op != "replace" else "replace_n"] += 1
    return holes, shape


BINDERS = {"let", "loop", "fn", "defn", "defn-", "for", "doseq", "binding", "when-let", "if-let", "letfn", "with-open", "dotimes"}


def roles_of(src: str) -> tuple[list[str], dict[str, set[str]]]:
    """A syntactic role per token (call = right after `(`, binding = inside the vector that follows a
    binder, key = keyword, else arg) and, per token text, every role it is seen in. This is the
    proxy for type pruning: these pairs are .cljc, and kotoba-sema types .kotoba sources only."""
    ts = tokens(src)
    roles, seen = [], {}
    stack: list[str] = []  # "bind" while inside a binder's vector, else "vec"/"paren"
    prev = None
    for t in ts:
        if t == "[":
            stack.append("bind" if prev in BINDERS else "vec")
            roles.append("punct")
        elif t in "({":
            stack.append("paren")
            roles.append("punct")
        elif t in ")]}":
            if stack:
                stack.pop()
            roles.append("punct")
        else:
            if t.startswith(":"):
                r = "key"
            elif prev == "(":
                r = "call"
            elif stack and stack[-1] == "bind":
                r = "binding"
            else:
                r = "arg"
            roles.append(r)
            seen.setdefault(t, set()).add(r)
        prev = t
    return roles, seen


def repo_sources(repo: str, sha: str) -> str:
    """Every .clj/.cljc/.cljk/.kotoba under src/ and test/ at `sha`, concatenated — the pool a
    symbol-index over the repo would offer."""
    r = subprocess.run(["git", "ls-tree", "-r", "--name-only", sha, "--", "src", "test"], cwd=repo, capture_output=True, text=True)
    out = []
    for f in r.stdout.split():
        if f.endswith((".clj", ".cljc", ".cljk", ".kotoba")):
            out.append(subprocess.run(["git", "show", f"{sha}:{f}"], cwd=repo, capture_output=True, text=True).stdout)
    return "\n".join(out)


def options_for(hole: dict, before: str, test: str, extra: str = "", prune_role: bool = False, cap: int = 255) -> tuple[list[str], dict]:
    pool_src = before + "\n" + test + ("\n" + extra if extra else "")
    pool = [t for t in tokens(pool_src) if kind_of(t) == hole["kind"]]
    if hole["kind"] == "str":
        # a name string is usually the name of a keyword nearby: offer every keyword's name as a string too
        pool += ['"' + t[1:] + '"' for t in tokens(pool_src) if kind_of(t) == "keyword"]
    seen, opts = {hole["old"]}, []
    for t in pool:
        if t not in seen:
            seen.add(t)
            opts.append(t)
    info = {"pool": len(opts), "gold_in_pool": hole["gold"] in opts}
    if prune_role:
        roles, seen_roles = roles_of(pool_src)
        hole_role = roles_of(before)[0][hole["pos"]]
        opts = [t for t in opts if hole_role in seen_roles.get(t, set())]
        info.update({"role": hole_role, "after_role": len(opts), "gold_after_role": hole["gold"] in opts})
    if len(opts) > cap:
        old = hole["old"]
        grams = {old[i:i + 3] for i in range(max(1, len(old) - 2))}
        opts.sort(key=lambda t: -len(grams & {t[i:i + 3] for i in range(max(1, len(t) - 2))}))
        opts = opts[:cap]
    info["after_cap"] = len(opts)
    return opts, info


def with_hole(before: str, pos: int, marker: str = "<<HOLE>>") -> str:
    # re-emit by replacing the pos-th token occurrence in the original text, preserving layout
    out, idx, i = [], 0, 0
    for m in TOK.finditer(before):
        out.append(before[i:m.start()])
        out.append(marker if idx == pos else m.group(0))
        i = m.end()
        idx += 1
    out.append(before[i:])
    return "".join(out)


def apply_fills(before: str, fills: dict[int, str]) -> str:
    out, idx, i = [], 0, 0
    for m in TOK.finditer(before):
        out.append(before[i:m.start()])
        out.append(fills.get(idx, m.group(0)))
        i = m.end()
        idx += 1
    out.append(before[i:])
    return "".join(out)


def openrouter_key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    out = subprocess.run(["security", "find-generic-password", "-s", "gftd.openrouter", "-a", "OPENROUTER_API_KEY", "-w"],
                         capture_output=True, text=True)
    k = out.stdout.strip()
    if not k or "could not" in k:
        raise SystemExit("REFUSED\topenrouter-key-missing")
    return k


def jev_choice(state: str, instructions: str, options: list[str], model: str, key: str) -> dict:
    body = json.dumps({"model": model, "state": state,
                       "questions": {"fill": {"type": "choice", "instructions": instructions,
                                              "criteria": {o: o for o in options}}}}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/alpha/decisions", data=body, method="POST",
                                 headers={"authorization": f"Bearer {key}", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    a = d["answers"]["fill"]
    return {"choice": a["choice"], "probabilities": a["probabilities"], "confidence": a.get("confidence"),
            "cost": d.get("usage", {}).get("cost"), "model": d.get("model")}


def ranked(probs: dict[str, float]) -> list[str]:
    return [k for k, _ in sorted(probs.items(), key=lambda kv: -kv[1])]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/kotoba-tasks.json")
    ap.add_argument("--top", default="../../..", help="superproject root holding orgs/kotoba-lang/<repo>")
    ap.add_argument("--model", default=os.environ.get("JEV_MODEL", "typesafe/jev-1.13"))
    ap.add_argument("--out", default="reports/jev-holes.json")
    ap.add_argument("--budget", type=int, default=8, help="max kbb runs per task incl. the greedy one")
    ap.add_argument("--only", default="", help="comma-separated task ids")
    ap.add_argument("--dry", action="store_true", help="no API, no kbb: just the corpus shape")
    ap.add_argument("--pool", choices=["file", "repo"], default="file", help="candidate source: module+test, or every source file in the repo at the sha")
    ap.add_argument("--prune-role", action="store_true", help="keep only candidates seen in the hole's syntactic role (proxy for type pruning)")
    a = ap.parse_args(argv)

    tasks = json.load(open(a.tasks))
    if a.only:
        keep = set(a.only.split(","))
        tasks = [t for t in tasks if t["id"] in keep]
    key = "" if a.dry else openrouter_key()
    report = {"model": a.model, "pool": a.pool, "prune_role": a.prune_role, "tasks": [], "corpus": {}}
    corpus = {"tasks": len(tasks), "with_holes": 0, "holes": 0, "ins_tokens": 0, "del_tokens": 0, "replace_n": 0, "insert": 0, "delete": 0}
    t_start = time.time()
    total_cost = 0.0

    for t in tasks:
        holes, shape = holes_of(t["src_before"], t["src_after"])
        for k in ("ins_tokens", "del_tokens", "replace_n", "insert", "delete"):
            corpus[k] += shape[k]
        corpus["holes"] += len(holes)
        row = {"id": t["id"], "repo": t["repo"], "message": t["message"], "shape": shape, "n_holes": len(holes), "holes": []}
        if holes:
            corpus["with_holes"] += 1
        if a.dry or not holes:
            report["tasks"].append(row)
            print(json.dumps({k: row[k] for k in ("id", "n_holes", "shape")}), flush=True)
            continue

        repo = os.path.join(a.top, "orgs", "kotoba-lang", t["repo"])
        extra = repo_sources(repo, t["sha"]) if a.pool == "repo" else ""
        for h in holes:
            opts, pinfo = options_for(h, t["src_before"], t["test"], extra, a.prune_role)
            reachable = h["gold"] in opts
            hrow = {**h, "n_options": len(opts), "reachable": reachable, "chance": 1.0 / len(opts) if opts else None, "pool_info": pinfo}
            if reachable:
                state = (f"Commit intent: {t['message']}\n\n"
                         f"Test namespace (this is the specification the module must satisfy):\n{t['test']}\n\n"
                         f"Module {t['module_ns']} with exactly one token replaced by <<HOLE>>:\n{with_hole(t['src_before'], h['pos'])}\n\n"
                         f"Test output before the fix:\n{t['before_tail']}")
                instr = f"Which {h['kind']} should replace <<HOLE>> so that the test namespace passes?"
                try:
                    ans = jev_choice(state, instr, opts, a.model, key)
                    order = ranked(ans["probabilities"])
                    hrow.update({"top1": ans["choice"], "correct": ans["choice"] == h["gold"],
                                 "gold_rank": order.index(h["gold"]) + 1 if h["gold"] in order else None,
                                 "confidence": ans["confidence"], "p_gold": ans["probabilities"].get(h["gold"]),
                                 "order": order[:10], "cost": ans["cost"]})
                    total_cost += ans["cost"] or 0.0
                except Exception as e:  # noqa: BLE001
                    hrow.update({"error": str(e)[:200]})
            row["holes"].append(hrow)
            print(json.dumps({k: hrow.get(k) for k in ("old", "gold", "n_options", "reachable", "top1", "correct", "gold_rank", "confidence", "pool_info")}), flush=True)

        # end-to-end: greedy fill, then backtrack on the model's own ranking
        runs = []
        answered = [h for h in row["holes"] if "order" in h]
        partial = len(answered) < len(holes)
        if answered:
            choice_idx = {h["pos"]: 0 for h in answered}
            by_conf = sorted(answered, key=lambda h: (h["confidence"] or 0.0))
            verdict = None
            for run_i in range(a.budget):
                fills = {h["pos"]: h["order"][min(choice_idx[h["pos"]], len(h["order"]) - 1)] for h in answered}
                src = apply_fills(t["src_before"], fills)
                v, tail = run_case(repo, t["sha"], t["src_path"], src, t["test_ns"], "kbb")
                runs.append({"run": run_i, "fills": {str(k): v_ for k, v_ in fills.items()}, "verdict": v, "tail": tail[-200:]})
                print(json.dumps({"id": t["id"], "run": run_i, "verdict": v}), flush=True)
                if v == "PASS":
                    verdict = "PASS"
                    break
                # backtrack: advance the least-confident hole that still has a next option; reset none
                advanced = False
                for h in by_conf:
                    if choice_idx[h["pos"]] + 1 < len(h["order"]):
                        choice_idx[h["pos"]] += 1
                        advanced = True
                        break
                if not advanced:
                    break
            row["e2e"] = {"verdict": verdict or "FAIL", "runs": len(runs), "greedy_pass": bool(runs) and runs[0]["verdict"] == "PASS",
                          "partial": partial, "unreachable_left_as_is": len(holes) - len(answered)}
        else:
            row["e2e"] = {"verdict": "SKIPPED", "reason": "no hole answered", "runs": 0}
        row["runs"] = runs
        report["tasks"].append(row)
        print(json.dumps({"id": t["id"], "e2e": row["e2e"]}), flush=True)

    answered = [h for r in report["tasks"] for h in r.get("holes", []) if "correct" in h]
    reachable = [h for r in report["tasks"] for h in r.get("holes", []) if h.get("reachable") is not None]
    report["corpus"] = corpus
    report["summary"] = {
        "holes_total": corpus["holes"],
        "holes_reachable": sum(1 for h in reachable if h["reachable"]),
        "holes_answered": len(answered),
        "top1_acc": (sum(h["correct"] for h in answered) / len(answered)) if answered else None,
        "mean_chance": (sum(h["chance"] for h in answered) / len(answered)) if answered else None,
        "mrr": (sum(1.0 / h["gold_rank"] for h in answered if h["gold_rank"]) / len(answered)) if answered else None,
        "top3_acc": (sum(1 for h in answered if h["gold_rank"] and h["gold_rank"] <= 3) / len(answered)) if answered else None,
        "mean_confidence_correct": (sum(h["confidence"] or 0 for h in answered if h["correct"]) / max(1, sum(h["correct"] for h in answered))) if answered else None,
        "mean_confidence_wrong": (sum(h["confidence"] or 0 for h in answered if not h["correct"]) / max(1, sum(not h["correct"] for h in answered))) if answered else None,
        "tasks_attempted_e2e": sum(1 for r in report["tasks"] if r.get("e2e", {}).get("runs", 0) > 0),
        "tasks_pass_greedy": sum(1 for r in report["tasks"] if r.get("e2e", {}).get("greedy_pass")),
        "tasks_pass_backtrack": sum(1 for r in report["tasks"] if r.get("e2e", {}).get("verdict") == "PASS"),
        "kbb_runs": sum(r.get("e2e", {}).get("runs", 0) for r in report["tasks"]),
        "api_cost_usd": round(total_cost, 6),
        "wall_s": round(time.time() - t_start, 1),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(report, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({"corpus": corpus, "summary": report["summary"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
