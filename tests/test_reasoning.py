"""Tests for reasoning extraction (readable explanations + citations)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.analysis import reasoning as R
from src.sandbox import db as dbm


def _insert_run(conn, *, agent="long", status="ok", response="bought stuff",
                decisions=None, tools_called=None, error=None):
    conn.execute(
        "INSERT INTO agent_runs(ts, agent, status, prompt, response, tools_called, "
        "decisions, error, latency_ms) VALUES (?, ?, ?, 'p', ?, ?, ?, ?, 10)",
        (datetime.now(timezone.utc).isoformat(), agent, status, response,
         json.dumps(tools_called) if tools_called is not None else None,
         json.dumps(decisions) if decisions is not None else None, error),
    )
    return conn.execute("SELECT * FROM agent_runs ORDER BY id DESC LIMIT 1").fetchone()


def test_decisions_normalized():
    raw = [
        {"symbol": "ORCL", "side": "buy", "qty": 99, "thesis": "strong buy",
         "accepted": True, "reject_reason": None},
        {"symbol": "XYZ", "side": "buy", "qty": 0, "thesis": "",
         "accepted": False, "reject_reason": "size_rounded_to_zero"},
    ]
    out = R.decisions(raw)
    assert out[0]["action"] == "Buy"
    assert out[0]["outcome"] == "Placed"
    assert out[1]["outcome"] == "Skipped (size_rounded_to_zero)"


def test_data_sources_extracts_citations():
    tools = [
        {"step": 1, "tool_calls": [
            {"name": "get_recommendations", "args": {"top_n": 7},
             "result": {"count": 2, "rows": [{"ticker": "ORCL"}, {"ticker": "AVGO"}]}},
            {"name": "propose_rebalance", "args": {}, "result": {"ok": True}},
        ]},
        {"step": 2, "tool_calls": [
            {"name": "lookup_ticker", "args": {"symbol": "ORCL"},
             "result": {"symbol": "ORCL", "price": 140.64}},
        ]},
    ]
    src = R.data_sources(tools)
    # propose_rebalance is excluded; two read tools remain.
    names = [s["tool"] for s in src]
    assert names == ["get_recommendations", "lookup_ticker"]
    assert "ORCL" in src[0]["summary"]
    assert src[1]["summary"] == "ORCL @ 140.64"
    assert src[0]["label"] == "Long-term stock recommendations"


def test_data_sources_handles_error_result():
    tools = [{"step": 1, "tool_calls": [
        {"name": "list_intraday_ideas", "args": {}, "result": {"error": "boom"}}]}]
    src = R.data_sources(tools)
    assert src[0]["summary"] == "error: boom"


def test_explain_run_full(tmp_db):
    row = _insert_run(
        tmp_db, agent="long", status="ok", response="Bought 5 names.",
        decisions=[{"symbol": "MSFT", "side": "buy", "qty": 36, "thesis": "quality",
                    "accepted": True}],
        tools_called=[{"step": 1, "tool_calls": [
            {"name": "get_recommendations", "args": {},
             "result": {"rows": [{"ticker": "MSFT"}]}}]}],
    )
    exp = R.explain_run(row)
    assert exp["agent"] == "long"
    assert exp["rationale"] == "Bought 5 names."
    assert exp["decisions"][0]["symbol"] == "MSFT"
    assert exp["data_sources"][0]["tool"] == "get_recommendations"


def test_learning_records_includes_no_trade(tmp_db):
    _insert_run(tmp_db, agent="day", status="ok", response="no trades today",
                decisions=[], tools_called=[{"step": 1, "tool_calls": [
                    {"name": "list_intraday_ideas", "args": {}, "result": {"rows": []}}]}])
    recs = R.learning_records(tmp_db)
    assert len(recs) == 1
    assert recs[0]["action"] == "no_trade"
    assert recs[0]["rationale"] == "no trades today"


def test_learning_records_one_per_decision(tmp_db):
    _insert_run(tmp_db, agent="long", response="ok",
                decisions=[{"symbol": "A", "side": "buy", "qty": 1, "accepted": True},
                           {"symbol": "B", "side": "buy", "qty": 2, "accepted": True}],
                tools_called=[])
    recs = R.learning_records(tmp_db)
    syms = sorted(r["symbol"] for r in recs)
    assert syms == ["A", "B"]


def test_append_run_writes_jsonl(tmp_path):
    path = tmp_path / "reasoning_log.jsonl"
    n = R.append_run(
        run_id=1, ts="2026-07-13T17:00:00+00:00", agent="day", status="ok",
        response="Bought AAPL and shorted TSLA.",
        decisions_obj=[
            {"symbol": "AAPL", "side": "buy", "qty": 10, "thesis": "breakout",
             "accepted": True},
            {"symbol": "TSLA", "side": "sell", "qty": 5, "thesis": "breakdown",
             "accepted": True},
        ],
        tools_called_obj=[{"step": 1, "tool_calls": [
            {"name": "list_intraday_ideas", "args": {},
             "result": {"rows": [{"symbol": "AAPL"}, {"symbol": "TSLA"}]}}]}],
        error=None, path=path,
    )
    assert n == 2
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    recs = [json.loads(ln) for ln in lines]
    assert {r["symbol"] for r in recs} == {"AAPL", "TSLA"}
    assert recs[0]["data_sources"][0]["tool"] == "list_intraday_ideas"


def test_append_run_never_raises(tmp_path):
    # Malformed inputs must not raise (best-effort logging).
    path = tmp_path / "sub" / "log.jsonl"
    n = R.append_run(run_id=None, ts="", agent="x", status="ok", response=None,
                     decisions_obj="not-json", tools_called_obj=None,
                     error=None, path=path)
    assert n >= 0
