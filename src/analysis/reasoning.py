"""Reasoning extraction — turn raw `agent_runs` rows into readable, structured
explanations, and build a durable learning dataset.

Everything the agents "think" is already persisted in `agent_runs`:
- `response`      — the LLM's natural-language rationale
- `decisions`     — JSON list of sized/validated trade decisions (with thesis)
- `tools_called`  — JSON step log of every tool call (name, args, result)

The tool-call log doubles as the **data-source citation trail**: each read tool
(e.g. `get_recommendations`, `list_intraday_ideas`, `lookup_ticker`, `get_news`)
records exactly what upstream data informed a decision. This module parses those
into a compact, human-readable form for the dashboard and for offline learning.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

# Tools that *read* data (as opposed to buffering a proposed trade). Their
# results are the citations behind a decision.
READ_TOOLS = {
    "get_recommendations", "get_portfolio_suggestion", "list_intraday_ideas",
    "get_quote", "lookup_ticker", "get_news", "current_positions",
    "account_snapshot", "scan_latest", "market_status",
}
PROPOSE_TOOLS = {"propose_trade", "propose_rebalance"}

_TOOL_LABEL = {
    "get_recommendations": "Long-term stock recommendations",
    "get_portfolio_suggestion": "Suggested portfolio weights",
    "list_intraday_ideas": "Intraday trade ideas (scanner)",
    "get_quote": "Live quote / recent bars",
    "lookup_ticker": "Ticker fundamentals & price",
    "get_news": "Recent news headlines",
    "current_positions": "Current holdings",
    "account_snapshot": "Account equity & cash",
    "scan_latest": "Latest market scan",
    "market_status": "Market open/close status",
}


def _loads(raw: Any) -> Any:
    if not raw:
        return None
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _summarize_result(name: str, result: Any) -> str:
    """One-line, human-readable summary of a tool result for citation display."""
    if result is None:
        return "no data returned"
    if isinstance(result, dict):
        # Common shapes: {count, rows:[...]} / {recommendations:[...]} / a quote dict
        for key in ("rows", "recommendations", "ideas", "suggestion", "news"):
            if key in result and isinstance(result[key], list):
                items = result[key]
                syms = [str(it.get("ticker") or it.get("symbol") or "")
                        for it in items if isinstance(it, dict)]
                syms = [s for s in syms if s]
                head = ", ".join(syms[:6])
                more = f" (+{len(syms) - 6} more)" if len(syms) > 6 else ""
                return f"{len(items)} item(s)" + (f": {head}{more}" if head else "")
        if "error" in result:
            return f"error: {result['error']}"
        # A single-symbol quote/lookup
        sym = result.get("symbol") or result.get("ticker")
        price = result.get("price") or result.get("last")
        if sym and price is not None:
            return f"{sym} @ {price}"
        # Fall back to a few keys
        keys = ", ".join(list(result.keys())[:5])
        return f"{{{keys}}}"
    if isinstance(result, list):
        return f"{len(result)} item(s)"
    return str(result)[:120]


def data_sources(tools_called: Any) -> list[dict[str, str]]:
    """Extract the data sources (read-tool calls) that informed a run.

    Returns a list of {tool, label, query, summary} — the citation trail.
    """
    steps = _loads(tools_called) or []
    out: list[dict[str, str]] = []
    for step in steps:
        for call in (step.get("tool_calls") or []) if isinstance(step, dict) else []:
            name = call.get("name", "")
            if name not in READ_TOOLS:
                continue
            out.append({
                "tool": name,
                "label": _TOOL_LABEL.get(name, name),
                "query": json.dumps(call.get("args") or {}, separators=(",", ":")),
                "summary": _summarize_result(name, call.get("result")),
            })
    return out


def decisions(decisions_json: Any) -> list[dict[str, Any]]:
    """Normalize the decisions list into readable action records."""
    raw = _loads(decisions_json) or []
    out: list[dict[str, Any]] = []
    for d in raw:
        if not isinstance(d, dict):
            continue
        accepted = bool(d.get("accepted"))
        side = (d.get("side") or "").lower()
        action = {"buy": "Buy", "sell": "Sell"}.get(side, side or "?")
        out.append({
            "symbol": d.get("symbol", "?"),
            "action": action,
            "qty": d.get("qty", 0),
            "thesis": d.get("thesis") or "",
            "accepted": accepted,
            "outcome": "Placed" if accepted else f"Skipped ({d.get('reject_reason') or 'n/a'})",
        })
    return out


def explain_run(row: sqlite3.Row) -> dict[str, Any]:
    """Full structured explanation of one agent run for display."""
    return {
        "id": row["id"],
        "ts": row["ts"],
        "agent": row["agent"],
        "status": row["status"],
        "rationale": (row["response"] or "").strip(),
        "decisions": decisions(row["decisions"]),
        "data_sources": data_sources(row["tools_called"]),
        "error": row["error"],
    }


def learning_records(conn: sqlite3.Connection, *, limit: int = 1000) -> list[dict[str, Any]]:
    """Flatten runs → per-decision records for an offline learning dataset.

    Each record pairs the agent's reasoning at decision time with the data it
    cited, so a future model can learn which reasoning/data led to good trades.
    """
    rows = conn.execute(
        "SELECT id, ts, agent, status, response, decisions, tools_called, error "
        "FROM agent_runs ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    records: list[dict[str, Any]] = []
    for row in rows:
        exp = explain_run(row)
        sources = exp["data_sources"]
        if not exp["decisions"]:
            # Still record no-trade reasoning — useful negative examples.
            records.append({
                "run_id": exp["id"], "ts": exp["ts"], "agent": exp["agent"],
                "symbol": None, "action": "no_trade", "qty": 0,
                "thesis": "", "accepted": False,
                "rationale": exp["rationale"], "data_sources": sources,
            })
            continue
        for d in exp["decisions"]:
            records.append({
                "run_id": exp["id"], "ts": exp["ts"], "agent": exp["agent"],
                "symbol": d["symbol"], "action": d["action"], "qty": d["qty"],
                "thesis": d["thesis"], "accepted": d["accepted"],
                "rationale": exp["rationale"], "data_sources": sources,
            })
    return records
