import json
import sqlite3
import os

from typed_decisions.hermes_data import scrub, outcome_of, extract_db


def test_scrub_and_outcome():
    s, n = scrub("Authorization: Bearer sk-or-v1-abcdefghijklmnop and KOTOBA_API_TOKEN=abc123 plus " + "a" * 70)
    assert "[REDACTED]" in s and "sk-or" not in s and "abc123" not in s and n >= 3
    assert outcome_of("terminal", '{"output": "", "exit_code": 0, "error": null}') == 1
    assert outcome_of("terminal", '{"output": "x", "exit_code": 2, "error": null}\n\n[Subdirectory context]') == 0
    assert outcome_of("read_file", "Traceback (most recent call last)") == 0
    assert outcome_of("read_file", "ok contents") == 1
    assert outcome_of("read_file", None) is None


def test_extract_from_synthetic_db(tmp_path):
    p = tmp_path / "state.db"
    con = sqlite3.connect(p)
    con.executescript("""
    create table sessions(id text primary key, cwd text, model text, tool_call_count int, started_at real);
    create table messages(id integer primary key autoincrement, session_id text, role text, content text, tool_calls text, tool_name text, tool_call_id text);
    insert into sessions values('s1','/w','m',2,1.0);
    insert into messages(session_id,role,content) values('s1','user','fix the test');
    insert into messages(session_id,role,tool_calls) values('s1','assistant','[{"id":"c1","function":{"name":"read_file","arguments":"{}"}}]');
    insert into messages(session_id,role,content,tool_name,tool_call_id) values('s1','tool','contents','read_file','c1');
    insert into messages(session_id,role,tool_calls) values('s1','assistant','[{"id":"c2","function":{"name":"terminal","arguments":"{}"}}]');
    insert into messages(session_id,role,content,tool_name,tool_call_id) values('s1','tool','{"output":"","exit_code":1,"error":null}','terminal','c2');
    """)
    con.commit()
    con.close()
    ex, st = extract_db(str(p), "prof")
    assert st["records"] == 2 and st["calls-with-outcome"] == 2
    e2 = ex[1]
    assert e2.questions[0].options == ["read_file", "terminal"] and e2.questions[0].gold == 1
    assert e2.questions[1].gold == 0  # exit_code 1
    assert "[read_file] contents" in e2.state and "terminal" not in e2.state.split("user:")[1].split("[read_file]")[0]
