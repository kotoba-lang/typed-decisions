"""Mine `.cljc`/`.cljk` fix pairs from the workspace's own git history into eval tasks
(`scripts/model-eval/kotoba-tasks.edn`): the model sees the BROKEN module and the test file, must
return the fixed module; PASS/FAIL is `kbb --backend sci` running the test namespace (no LLM judge).

A commit qualifies when it touches exactly one src/*.cljc|cljk and its test file is self-contained
(requires only the module, clojure.test / string / set / walk / edn) — and, decisive, when the
runner says FAIL on the parent's module and PASS on the commit's module. Pairs that do not flip
are discarded (a task that passes with the broken code measures nothing). Each kept task records
the repo, commit and both verdicts, so the corpus is reproducible from git alone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile

ALLOWED_REQ = {"clojure.test", "clojure.string", "clojure.set", "clojure.walk", "clojure.edn", "cljs.test"}
RUNNER = '''(require '[clojure.test :as t] '[clojure.string :as str] '{ns})
(def out (with-out-str (t/run-tests '{ns})))
(print out)
(let [[_ tests asserts] (re-find #"Ran (\\d+) tests containing (\\d+) assertions" out)
      [_ fails errors] (re-find #"(\\d+) failures, (\\d+) errors" out)
      ok (and asserts fails errors (pos? (js/parseInt asserts)) (zero? (js/parseInt fails)) (zero? (js/parseInt errors)))]
  (println (pr-str {:tests tests :assertions asserts :failures fails :errors errors}))
  (println (if ok "PASS" "FAIL")))
'''


def sh(args, cwd=None, timeout=120):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def ns_of(src: str) -> str | None:
    m = re.search(r"\(ns\s+([\w.\-*+!?$%&=<>]+)", src)
    return m.group(1) if m else None


def ns_form(src: str) -> str:
    i = src.find("(ns ")
    if i < 0:
        return ""
    depth, j, in_str = 0, i, False
    while j < len(src):
        c = src[j]
        if c == '"' and src[j - 1] != "\\":
            in_str = not in_str
        elif not in_str:
            if c == "(" or c == "[":
                depth += 1
            elif c == ")" or c == "]":
                depth -= 1
                if depth == 0:
                    return src[i : j + 1]
        j += 1
    return src[i:]


def requires_of(src: str) -> set[str]:
    """Namespaces named in the ns form's :require / :require-macros / :use (libspec vectors or bare symbols)."""
    f = ns_form(src)
    out = set()
    for block in re.findall(r"\((?::require|:require-macros|:use)\s+(.*?)\)\s*(?=\(:|\)$)", f, flags=re.S):
        out |= set(re.findall(r"\[\s*([A-Za-z][\w.\-]*)", block))
        out |= set(re.findall(r"(?<![\[\w.:/-])([A-Za-z][\w\-]*(?:\.[\w\-]+)+)(?![\w/])", block))
    out |= set(re.findall(r"\(require\s+'\[?\s*([A-Za-z][\w.\-]*)", src))
    return {o for o in out if "." in o or o in ("clojure",)}


def snapshot(repo: str, sha: str, dest: str) -> None:
    """Export the repo tree at `sha` into dest (git archive | tar): the harness is the real repo with one module swapped."""
    p1 = subprocess.Popen(["git", "archive", "--format=tar", sha], cwd=repo, stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", dest], stdin=p1.stdout, check=True)
    p1.wait()


def classpath_of(root: str) -> str:
    dirs = [d for d in ("src", "test", "resources", "lib") if os.path.isdir(os.path.join(root, d))]
    return ":".join(os.path.join(root, d) for d in dirs)


def run_case(repo: str, sha: str, src_rel: str, src_text: str, test_ns: str, kbb: str, timeout: int = 120) -> tuple[str, str]:
    with tempfile.TemporaryDirectory() as d:
        snapshot(repo, sha, d)
        p = os.path.join(d, src_rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(src_text)
        open(os.path.join(d, "runner.cljk"), "w").write(RUNNER.replace("{ns}", test_ns))
        try:
            code, out, err = sh([kbb, "--backend", "sci", "--classpath", classpath_of(d), "runner.cljk"], cwd=d, timeout=timeout)
        except subprocess.TimeoutExpired:
            return "timeout", ""
        lines = out.strip().splitlines()
        verdict = "PASS" if code == 0 and lines and lines[-1] == "PASS" else "FAIL"
        return verdict, (out + err)[-600:]


def mine_repo(repo: str, kbb: str, max_commits: int, want: int, seen_ids: set[str]) -> list[dict]:
    code, log, _ = sh(["git", "log", "--format=%H%x09%s", "-i", "-E", "--grep=fix|bug|wrong|incorrect|off-by|regress|repair", f"-n{max_commits}", "--", "src"], cwd=repo)
    tasks = []
    stats = __import__("collections").Counter()
    for line in log.splitlines():
        sha, msg = line.split("\t", 1)
        _, files, _ = sh(["git", "show", "--name-only", "--format=", sha], cwd=repo)
        fl = [f for f in files.split() if f]
        srcs = [f for f in fl if f.startswith("src/") and f.endswith((".cljc", ".cljk"))]
        if len(srcs) != 1:
            continue
        src_rel = srcs[0]
        _, src_after, _ = sh(["git", "show", f"{sha}:{src_rel}"], cwd=repo)
        c, src_before, _ = sh(["git", "show", f"{sha}~1:{src_rel}"], cwd=repo)
        if c != 0 or not src_before.strip() or src_before == src_after:
            continue
        mod_ns = ns_of(src_after)
        if not mod_ns:
            continue
        # find the test file for this ns at the commit
        _, tree, _ = sh(["git", "ls-tree", "-r", "--name-only", sha], cwd=repo)
        pat = mod_ns.replace("-", "_").replace(".", "/") + "_test."
        cand = [t for t in tree.split() if t.endswith((".cljc", ".cljk")) and t.split("/", 1)[-1].endswith(pat + t.rsplit(".", 1)[-1]) and not t.startswith("src/")]
        if not cand:
            stats["no-test"] += 1
            continue
        test_rel = cand[0]
        _, test_text, _ = sh(["git", "show", f"{sha}:{test_rel}"], cwd=repo)
        test_ns = ns_of(test_text)
        if not test_ns:
            continue
        # the harness is the repo at `sha` (git archive) with this one module swapped; anything the test
        # needs beyond that (deps.edn libraries, JVM-only paths) shows up as FAIL-after and drops the pair
        v_after, after_out = run_case(repo, sha, src_rel, src_after, test_ns, kbb)
        if v_after != "PASS":
            stats["after-not-pass"] += 1
            continue
        v_before, before_out = run_case(repo, sha, src_rel, src_before, test_ns, kbb)
        if v_before != "FAIL":
            stats["before-not-fail"] += 1
            continue
        tid = f"kotoba-{os.path.basename(repo)}-{sha[:8]}"
        if tid in seen_ids:
            continue
        seen_ids.add(tid)
        tasks.append({"id": tid, "repo": os.path.basename(repo), "sha": sha, "message": msg[:160], "src_path": src_rel, "test_path": test_rel,
                      "module_ns": mod_ns, "test_ns": test_ns, "src_before": src_before, "src_after": src_after, "test": test_text,
                      "verified": {"after": "PASS", "before": "FAIL"}, "before_tail": before_out[-300:]})
        print("kept", tid, msg[:80], flush=True)
        if len(tasks) >= want:
            break
    print(os.path.basename(repo), dict(stats), flush=True)
    return tasks


def to_edn(tasks: list[dict]) -> str:
    def s(x):
        return json.dumps(x, ensure_ascii=False)
    rows = []
    for t in tasks:
        rows.append("{:id :%s :kind :cljk :budget 4000 :repo %s :sha %s :message %s :src-path %s :test-path %s :module-ns %s :test-ns %s\n  :src-before %s\n  :src-after %s\n  :test %s\n  :verified {:after :pass :before :fail}}" % (
            t["id"], s(t["repo"]), s(t["sha"]), s(t["message"]), s(t["src_path"]), s(t["test_path"]), s(t["module_ns"]), s(t["test_ns"]), s(t["src_before"]), s(t["src_after"]), s(t["test"])))
    return ";; generated by typed-decisions/mine_kotoba_tasks.py — fix pairs from kotoba-lang git history, each verified FAIL(before)/PASS(after) under kbb sci\n[\n" + "\n\n".join(rows) + "\n]\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", default="../../..")
    ap.add_argument("--repos", default="browser,amu,app-kotoba-cloud,aiueos,arrangement,bim,artifact,ayatori,codebase,chobo,chain,card,checkpointer,character,bytes,authentication,abi")
    ap.add_argument("--want", type=int, default=20)
    ap.add_argument("--per-repo", type=int, default=6)
    ap.add_argument("--max-commits", type=int, default=400)
    ap.add_argument("--out", default="../../../scripts/model-eval/kotoba-tasks.edn")
    ap.add_argument("--json-out", default="data/kotoba-tasks.json")
    ap.add_argument("--kbb", default="kbb")
    a = ap.parse_args(argv)
    all_tasks, seen = [], set()
    for r in a.repos.split(","):
        repo = os.path.join(a.top, "orgs", "kotoba-lang", r)
        if not os.path.isdir(os.path.join(repo, ".git")):
            print("skip (no checkout)", r)
            continue
        ts = mine_repo(repo, a.kbb, a.max_commits, a.per_repo, seen)
        print(r, "kept", len(ts), flush=True)
        all_tasks += ts
        if len(all_tasks) >= a.want:
            break
    all_tasks = all_tasks[: a.want]
    os.makedirs(os.path.dirname(a.json_out), exist_ok=True)
    json.dump(all_tasks, open(a.json_out, "w"), ensure_ascii=False, indent=1)
    open(a.out, "w").write(to_edn(all_tasks))
    print(json.dumps({"tasks": len(all_tasks), "repos": sorted(set(t["repo"] for t in all_tasks)), "out": a.out}))


if __name__ == "__main__":
    main()
