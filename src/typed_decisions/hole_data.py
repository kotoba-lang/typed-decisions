"""The `code-holes` family: single-token substitutions mined from kotoba-lang git history, with gold.

Every hole is a real decision a person made in a real commit — replace token A by token B in one
place — and the gold is B. That is stronger than a teacher label and weaker than the test-verified
pairs of `mine_kotoba_tasks.py` (which also live here, tagged `verified: "test"`): a commit can be
wrong, a test cannot pass by accident. Nothing is synthesised; no distractor is invented — the
options are the same-kind tokens the module (and its test, when one exists at that sha) already
contains, minus the token being replaced. A hole whose gold is not among those is counted and
dropped, not rescued.

    python -m typed_decisions.hole_data --top <superproject> --out data-holes [--repos N] [--max-commits N]

Writes train/val/test.jsonl in the `schema.Example` shape (split by repository, so test holes come
from repositories never seen in training), plus stats.json with what was mined, dropped and why.

Each record's meta carries `changed_tokens` (inserted + deleted tokens in that file's diff) and
`holes_in_file`. Measured 2026-09-19 on the first mining: 167/200 sampled holes sat in diffs of
>100 changed tokens — a 1:1 token alignment inside a rewrite, not a substitution anyone decided on —
and there the model scored 0.05 on symbols. Filter on `changed_tokens` before training or scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter

from .jev_holes import holes_of, kind_of, roles_of, tokens, with_hole
from .mine_kotoba_tasks import ns_of
from .schema import Example, Question, write_jsonl

SRC_EXT = (".clj", ".cljc", ".cljk", ".kotoba")


def sh(args, cwd):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stdout


def role_first_cap(opts: list[str], hole: dict, pool_src: str, cap: int) -> list[str]:
    """Rank candidates by (same syntactic role as the hole, char-3-gram overlap with the old token)
    and cut at `cap`. Role first, so the cap does not select for lookalikes alone (第7反復)."""
    if len(opts) <= cap:
        return opts
    _, seen = roles_of(pool_src)
    hole_role = roles_of(hole["_before"])[0][hole["pos"]]
    old = hole["old"]
    grams = {old[i:i + 3] for i in range(max(1, len(old) - 2))}
    opts.sort(key=lambda t: (-(hole_role in seen.get(t, set())), -len(grams & {t[i:i + 3] for i in range(max(1, len(t) - 2))})))
    return opts[:cap]


def options_for(hole: dict, before: str, test: str, cap: int = 255) -> list[str]:
    pool_src = before + "\n" + test
    pool = [t for t in tokens(pool_src) if kind_of(t) == hole["kind"]]
    if hole["kind"] == "str":
        pool += ['"' + t[1:] + '"' for t in tokens(pool_src) if kind_of(t) == "keyword"]
    seen, opts = {hole["old"]}, []
    for t in pool:
        if t not in seen:
            seen.add(t)
            opts.append(t)
    return role_first_cap(opts, {**hole, "_before": before}, pool_src, cap)


def test_for(repo: str, sha: str, mod_ns: str) -> str:
    _, tree = sh(["git", "ls-tree", "-r", "--name-only", sha], repo)
    pat = mod_ns.replace("-", "_").replace(".", "/") + "_test."
    for t in tree.split():
        if t.endswith(SRC_EXT) and not t.startswith("src/") and t.split("/", 1)[-1].endswith(pat + t.rsplit(".", 1)[-1]):
            return sh(["git", "show", f"{sha}:{t}"], repo)[1]
    return ""


def mine_repo(repo: str, max_commits: int, stats: Counter, max_files: int = 3) -> list[dict]:
    code, log = sh(["git", "log", "--format=%H%x09%s", f"-n{max_commits}", "--", "src"], repo)
    if code != 0:
        stats["repo-unreadable"] += 1
        return []
    out = []
    for line in log.splitlines():
        sha, msg = line.split("\t", 1)
        _, files = sh(["git", "show", "--name-only", "--format=", sha], repo)
        srcs = [f for f in files.split() if f.startswith("src/") and f.endswith(SRC_EXT)]
        if not srcs or len(srcs) > max_files:
            stats["commit-skipped-files"] += 1
            continue
        for src_rel in srcs:
            _, after = sh(["git", "show", f"{sha}:{src_rel}"], repo)
            c, before = sh(["git", "show", f"{sha}~1:{src_rel}"], repo)
            if c != 0 or not before.strip() or before == after:
                stats["file-no-parent"] += 1
                continue
            holes, shape = holes_of(before, after)
            stats["files-diffed"] += 1
            stats["ins_tokens"] += shape["ins_tokens"]
            stats["del_tokens"] += shape["del_tokens"]
            if not holes:
                continue
            mod_ns = ns_of(after) or src_rel
            test = test_for(repo, sha, mod_ns)
            for h in holes:
                stats[f"hole-{h['kind']}"] += 1
                opts = options_for(h, before, test)
                if h["gold"] not in opts:
                    stats["hole-unreachable"] += 1
                    continue
                out.append({"repo": os.path.basename(repo), "sha": sha, "message": msg[:200], "src_path": src_rel,
                            "module_ns": mod_ns, "has_test": bool(test), "pos": h["pos"], "old": h["old"], "gold": h["gold"],
                            "changed_tokens": shape["ins_tokens"] + shape["del_tokens"], "holes_in_file": len(holes),
                            "kind": h["kind"], "options": opts, "state": (
                                f"Commit intent: {msg[:200]}\n\n"
                                + (f"Test namespace (the specification the module must satisfy):\n{test}\n\n" if test else "")
                                + f"Module {mod_ns} with exactly one token replaced by <<HOLE>>:\n{with_hole(before, h['pos'])}")})
    return out


def to_example(h: dict, verified: str) -> Example:
    q = Question(qid=f"hole-{h['repo']}-{h['sha'][:8]}-{h['pos']}", kind="choice",
                 instructions=f"Which {h['kind']} should replace <<HOLE>> so that the module does what the commit intends?",
                 options=h["options"], gold=h["options"].index(h["gold"]))
    return Example(state=h["state"], questions=[q], source="code-holes",
                   meta={"repo": h["repo"], "sha": h["sha"], "src_path": h["src_path"], "old": h["old"], "kind": h["kind"],
                         "has_test": h["has_test"], "verified": verified, "n_options": len(h["options"]),
                         "changed_tokens": h["changed_tokens"], "holes_in_file": h["holes_in_file"]})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", default="../../..")
    ap.add_argument("--out", default="data-holes")
    ap.add_argument("--repos", type=int, default=80, help="mine this many repos, by src commit count")
    ap.add_argument("--max-commits", type=int, default=300)
    ap.add_argument("--tasks", default="data/kotoba-tasks.json", help="test-verified pairs: their holes get verified=test")
    a = ap.parse_args(argv)
    org = os.path.join(a.top, "orgs", "kotoba-lang")
    cands = []
    for d in sorted(os.listdir(org)):
        repo = os.path.join(org, d)
        if os.path.isdir(os.path.join(repo, ".git")) and os.path.isdir(os.path.join(repo, "src")):
            c, n = sh(["git", "rev-list", "--count", "HEAD", "--", "src"], repo)
            if c == 0 and n.strip().isdigit():
                cands.append((int(n.strip()), repo))
    cands.sort(reverse=True)
    chosen = [r for _, r in cands[: a.repos]]
    print(json.dumps({"repos_available": len(cands), "repos_mined": len(chosen), "deepest": cands[0][0] if cands else 0}), flush=True)

    verified_shas = set()
    if os.path.exists(a.tasks):
        verified_shas = {t["sha"] for t in json.load(open(a.tasks))}

    stats = Counter()
    holes = []
    for repo in chosen:
        hs = mine_repo(repo, a.max_commits, stats)
        holes += hs
        print(json.dumps({"repo": os.path.basename(repo), "holes": len(hs)}), flush=True)

    # split by repository: a hole's repo decides its split, so test repos are unseen in training
    def bucket(repo: str) -> str:
        x = int(hashlib.sha256(repo.encode()).hexdigest(), 16) % 10
        return "test" if x < 2 else "val" if x == 2 else "train"

    splits = {"train": [], "val": [], "test": []}
    for h in holes:
        splits[bucket(h["repo"])].append(to_example(h, "test" if h["sha"] in verified_shas else "commit"))
    os.makedirs(a.out, exist_ok=True)
    for k, v in splits.items():
        write_jsonl(os.path.join(a.out, f"{k}.jsonl"), v)
    kinds = Counter(h["kind"] for h in holes)
    n_opts = [len(h["options"]) for h in holes]
    iso = {f"<={b}": sum(1 for h in holes if h["changed_tokens"] <= b) for b in (10, 30, 100)}
    summary = {"holes_kept": len(holes), "by_kind": dict(kinds), "by_split": {k: len(v) for k, v in splits.items()}, "isolated": iso,
               "repos_with_holes": len({h["repo"] for h in holes}), "with_test": sum(h["has_test"] for h in holes),
               "verified_test": sum(1 for h in holes if h["sha"] in verified_shas),
               "mean_options": sum(n_opts) / len(n_opts) if n_opts else None, "mined": dict(stats)}
    json.dump(summary, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
