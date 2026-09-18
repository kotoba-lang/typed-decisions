"""Decision -> code / tool call: the shape the ADR proposes, as a runnable mapping.

A trained DecisionEncoder answers `Choice` over candidate definitions (options are definitions:
name, namespace, docstring, content hash), `Noul` over admission questions, and typed argument
slots. This module turns one such answer set into an EDN proposal the host can check — it never
executes anything (agents propose; the governor authorises):

    {:proposal/kind   :wire-reference
     :definition      "isobmff.mux/ftyp"           ; the state
     :reference       {:fq "isobmff.bytes/wu32" :hash "4b26e9d632"}   ; the Choice's pointer
     :probabilities   {...}                        ; the whole distribution, not just argmax
     :confidence      0.83
     :admit?          {:noul 0.91 :threshold 0.8 :decision :autonomous}   ; Noul above threshold
     :memo-key        "<sha256 of state + question + option hashes>"}     ; same input -> no forward

`memo_key` is the unison-like part: the decision is a pure function of content-addressed inputs,
so its key is the hash of (state, question, option hashes) — the same shape as symbol-index's
closure hash, which is already the memo key for compile/test results in this workspace.
"""

from __future__ import annotations

import hashlib
import json

from .schema import Question, readout


def memo_key(state: str, q: Question, option_hashes: list[str]) -> str:
    h = hashlib.sha256()
    h.update(state.encode())
    h.update(b"\0" + q.instructions.encode())
    for oh in option_hashes:
        h.update(b"\0" + oh.encode())
    return h.hexdigest()


def proposal(state_fq: str, state: str, q: Question, probs: list[float], option_hashes: list[str], admit_noul: float | None = None, threshold: float = 0.8) -> dict:
    r = readout(q.kind, probs)
    k = r.get("choice")
    out = {"proposal/kind": "wire-reference" if q.kind == "choice" else q.kind, "definition": state_fq,
           "probabilities": {q.options[i]: round(p, 4) for i, p in enumerate(probs)}, "memo-key": memo_key(state, q, option_hashes)}
    if k is not None:
        out["reference"] = {"fq": q.options[k].split(" — ")[0], "hash": option_hashes[k]}
        out["confidence"] = round(r["confidence"], 4)
    if admit_noul is not None:
        out["admit?"] = {"noul": round(admit_noul, 4), "threshold": threshold, "decision": "autonomous" if admit_noul >= threshold else "escalate"}
    return out


def to_edn(p: dict) -> str:
    """Minimal EDN emitter for the proposal map (strings, numbers, nested maps, keywords as :k)."""
    def emit(v):
        if isinstance(v, dict):
            return "{" + " ".join(f":{k} {emit(x)}" for k, x in v.items()) + "}"
        if isinstance(v, str):
            return json.dumps(v, ensure_ascii=False)
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, list):
            return "[" + " ".join(emit(x) for x in v) + "]"
        raise TypeError(type(v))
    return emit(p)
