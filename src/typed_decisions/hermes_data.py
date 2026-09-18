"""Hermes transcripts -> typed-decision records (stage 1 of the growth plan, ADR-2609181715 follow-up).

Reads every `~/.hermes/**/state.db` read-only (`?immutable=1`) and turns each assistant tool call into
two decisions on the context that preceded it:

  choice  "Which tool should the agent call next?"   options = tools the session has used (+ the called one), gold = the call
  noul    "Will this call succeed without an error?" gold from the tool result: terminal exit_code == 0, or no error marker

State = profile · cwd · the last user message · the last two tool results (truncated), never the assistant's own text of the
same turn (it names the tool = leak). Records are the Example shape of this repo so the same trainer / metrics run on them.
Split by session hash. Everything is counted: sessions, calls, calls with a readable result, per-profile error rate — the
"verified triples per day" number the plan asks for, measured for the first time.

Provenance / secrets: transcripts can carry credentials (request dumps under ~/.hermes/sessions show
`Authorization: Bearer …`). Every state string goes through `scrub` (bearer tokens, `sk-…` keys, long hex/base64
runs, `KEY=value` env lines are replaced by `[REDACTED]`) and the scrub count is reported; a record whose state
still matches a secret shape after scrubbing is dropped, not written. The output stays local (data-hermes/ is
gitignored) — nothing here uploads. Run it yourself: `python -m typed_decisions.hermes_data --out data-hermes`.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sqlite3
from collections import Counter

from .schema import Example, Question, NOUL_OPTIONS, write_jsonl

ERROR_MARKERS = ("Traceback", "Error:", "error:", "command not found", "No such file", "FAIL", "fatal:", "Permission denied")
SECRET_PATTERNS = [
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\b(?:ghp|gho|github_pat|xoxb|xoxp|AKIA)[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\b[A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY|APIKEY|PRIVATE_KEY)[A-Z0-9_]*\s*[=:]\s*\S+"),
    re.compile(r"\b[A-Fa-f0-9]{40,}\b"),
    re.compile(r"\b[A-Za-z0-9+/]{64,}={0,2}\b"),
]


def scrub(s: str) -> tuple[str, int]:
    n = 0
    for p in SECRET_PATTERNS:
        s, k = p.subn("[REDACTED]", s)
        n += k
    return s, n


def outcome_of(tool_name: str, content: str | None) -> int | None:
    if content is None:
        return None
    c = content.strip()
    if tool_name == "terminal":
        try:
            j = json.loads(c.split("\n\n[")[0])
            if isinstance(j, dict) and "exit_code" in j:
                return int(j.get("exit_code") == 0 and not j.get("error"))
        except Exception:
            pass
    if c.startswith("{") and '"error"' in c[:200]:
        try:
            j = json.loads(c)
            if isinstance(j, dict):
                return int(not j.get("error"))
        except Exception:
            pass
    return int(not any(m in c[:2000] for m in ERROR_MARKERS))


def trunc(s: str | None, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n // 2] + " … " + s[-n // 2 :]


def extract_db(path: str, profile: str, max_sessions: int | None = None) -> tuple[list[Example], dict]:
    con = sqlite3.connect(f"file:{path}?immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    stats = Counter()
    out: list[Example] = []
    sess = con.execute("select id, cwd, model, tool_call_count from sessions where tool_call_count > 0 order by started_at").fetchall()
    if max_sessions:
        sess = sess[:max_sessions]
    for s in sess:
        stats["sessions"] += 1
        rows = con.execute("select id, role, content, tool_calls, tool_name, tool_call_id from messages where session_id=? order by id", (s["id"],)).fetchall()
        results_by_call = {}
        for r in rows:
            if r["role"] == "tool" and r["tool_call_id"]:
                results_by_call[r["tool_call_id"]] = (r["tool_name"], r["content"])
        tools_used = sorted({r["tool_name"] for r in rows if r["role"] == "tool" and r["tool_name"]})
        last_user, last_tools = "", []
        for r in rows:
            if r["role"] == "user":
                last_user = r["content"] or ""
            elif r["role"] == "tool":
                last_tools = (last_tools + [(r["tool_name"], r["content"] or "")])[-2:]
            elif r["role"] == "assistant" and r["tool_calls"]:
                try:
                    calls = json.loads(r["tool_calls"])
                except Exception:
                    stats["bad-tool-calls-json"] += 1
                    continue
                for c in calls:
                    fn = (c.get("function") or {}).get("name")
                    if not fn:
                        continue
                    stats["calls"] += 1
                    cid = c.get("call_id") or c.get("id")
                    res = results_by_call.get(cid)
                    y = outcome_of(fn, res[1]) if res else None
                    if y is None:
                        stats["calls-without-result"] += 1
                    else:
                        stats["calls-with-outcome"] += 1
                        stats[f"ok:{fn}"] += y
                        stats[f"n:{fn}"] += 1
                    opts = sorted(set(tools_used) | {fn})
                    if len(opts) < 2:
                        stats["single-tool-session"] += 1
                        continue
                    state = (f"profile: {profile}\ncwd: {s['cwd'] or ''}\nuser: {trunc(last_user, 600)}\n" +
                             "".join(f"[{t}] {trunc(x, 300)}\n" for t, x in last_tools))
                    state, k = scrub(state)
                    stats["scrubbed-secrets"] += k
                    if any(p.search(state) for p in SECRET_PATTERNS):
                        stats["dropped-secret-shape"] += 1
                        continue
                    key = hashlib.sha1(f"{path}:{s['id']}:{r['id']}:{cid}".encode()).hexdigest()[:12]
                    qs = [Question(f"hermes-{key}-tool", "choice", "Which tool should the agent call next?", opts, opts.index(fn))]
                    if y is not None:
                        qs.append(Question(f"hermes-{key}-ok", "noul", f"Will the next {fn} call succeed without an error?", list(NOUL_OPTIONS), y))
                    out.append(Example(state=state, source="hermes", meta={"profile": profile, "session": s["id"], "tool": fn, "model": s["model"]}, questions=qs))
                    stats["records"] += 1
    con.close()
    return out, dict(stats)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hermes", default=os.path.expanduser("~/.hermes"))
    ap.add_argument("--out", default="data-hermes")
    ap.add_argument("--max-sessions-per-db", type=int, default=0)
    ap.add_argument("--max-profiles", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.1)
    a = ap.parse_args(argv)
    dbs = [(os.path.join(a.hermes, "state.db"), "default")] + [(p, os.path.basename(os.path.dirname(p))) for p in sorted(glob.glob(os.path.join(a.hermes, "profiles", "*", "state.db")))]
    if a.max_profiles:
        dbs = dbs[: a.max_profiles + 1]
    all_ex, total = [], Counter()
    per_profile = {}
    for path, prof in dbs:
        try:
            ex, st = extract_db(path, prof, a.max_sessions_per_db or None)
        except sqlite3.Error as e:
            total["db-unreadable"] += 1
            print("unreadable", prof, str(e)[:80])
            continue
        all_ex += ex
        total.update({k: v for k, v in st.items() if not k.startswith(("ok:", "n:"))})
        n_term, ok_term = st.get("n:terminal", 0), st.get("ok:terminal", 0)
        per_profile[prof] = {"records": st.get("records", 0), "terminal_calls": n_term, "terminal_ok_rate": (ok_term / n_term) if n_term else None}
    rng = random.Random(0)
    train, test = [], []
    for e in all_ex:
        (test if int(hashlib.sha1(e.meta["session"].encode()).hexdigest()[:6], 16) % 1000 < a.test_frac * 1000 else train).append(e)
    rng.shuffle(train)
    rng.shuffle(test)
    os.makedirs(a.out, exist_ok=True)
    val, train = train[:400], train[400:]
    write_jsonl(os.path.join(a.out, "train.jsonl"), train)
    write_jsonl(os.path.join(a.out, "val.jsonl"), val)
    write_jsonl(os.path.join(a.out, "test.jsonl"), test)
    tools = Counter(e.meta["tool"] for e in all_ex)
    rep = {"dbs": len(dbs), "totals": dict(total), "train": len(train), "val": len(val), "test": len(test), "tool_distribution": dict(tools.most_common(12)),
           "majority_tool_rate": (tools.most_common(1)[0][1] / max(1, len(all_ex))) if tools else None,
           "top_profiles": sorted(per_profile.items(), key=lambda kv: -kv[1]["records"])[:10]}
    json.dump(rep, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
