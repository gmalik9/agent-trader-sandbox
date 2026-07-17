#!/usr/bin/env python3
"""End-of-day review — aggregate the trading day and append a `session_close`
entry to the journal, then print reflection prompts.

Run once after the close (positions flat), inside the scheduler container:

    docker compose exec -T scheduler python \
        .github/skills/autonomous-day-trader/scripts/end_of_day_review.py

It reads READ-ONLY from data/sandbox.sqlite (fills, decisions, equity, throttle)
and appends a structured review to data/trader_journal.jsonl. It does NOT decide
your lessons for you — after running it, YOU distil 1-3 durable lessons into
data/strategy_notes.md (that curated file is what the agent 'learns').
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

# Make `import src.*` and the sibling `journal_append` import work when this
# script is run by path (sys.path[0] is the script dir; add the app root too).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, *[".."] * 4))
for _p in (_SCRIPT_DIR, _APP_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from src.config import db_path
    DB = str(db_path())
except Exception:
    DB = "data/sandbox.sqlite"

# Reuse the journal writer so the schema stays consistent.
from journal_append import append_entry  # type: ignore  # noqa: E402


def _ro_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def build_review() -> dict:
    now = datetime.now(timezone.utc)
    day0 = now.strftime("%Y-%m-%d")
    c = _ro_conn()

    # --- day run outcomes ---
    runs = c.execute(
        "SELECT status, COUNT(*) n FROM agent_runs "
        "WHERE agent='day' AND ts > ? GROUP BY status", (day0,)
    ).fetchall()
    run_counts = {r["status"]: r["n"] for r in runs}
    throttled = c.execute(
        "SELECT COUNT(*) n FROM agent_runs WHERE agent='day' AND ts > ? "
        "AND error LIKE 'rate_limited%'", (day0,)
    ).fetchone()["n"]

    # --- decisions today ---
    dec_rows = c.execute(
        "SELECT decisions FROM agent_runs WHERE agent='day' AND ts > ?", (day0,)
    ).fetchall()
    proposals = accepted = rejected = exits = 0
    reject_reasons: dict[str, int] = {}
    for r in dec_rows:
        try:
            ds = json.loads(r["decisions"] or "[]")
        except Exception:
            ds = []
        for d in ds:
            if not isinstance(d, dict):
                continue
            if "agent_exit" in d or "stop_exit" in d:
                exits += 1
            elif "symbol" in d:
                proposals += 1
                if d.get("accepted"):
                    accepted += 1
                else:
                    rejected += 1
                    rr = str(d.get("reject_reason") or "unknown")
                    reject_reasons[rr] = reject_reasons.get(rr, 0) + 1

    # --- fills today (Alpaca leg = source of truth) ---
    fills = c.execute(
        "SELECT symbol, side, qty, fill_price FROM orders "
        "WHERE venue='alpaca_paper' AND status='filled' AND ts > ?", (day0,)
    ).fetchall()
    n_fills = len(fills)
    buy_notional = sum((f["qty"] or 0) * (f["fill_price"] or 0)
                       for f in fills if f["side"] == "buy")
    sell_notional = sum((f["qty"] or 0) * (f["fill_price"] or 0)
                        for f in fills if f["side"] == "sell")
    traded_syms = sorted({f["symbol"] for f in fills})

    # --- realized P&L for the day account (best-effort) ---
    realized = None
    try:
        from src.analysis import pnl as pnl_mod
        aid_row = c.execute(
            "SELECT id FROM accounts WHERE name IN ('day_alpaca','day') "
            "ORDER BY (name='day_alpaca') DESC LIMIT 1"
        ).fetchone()
        if aid_row:
            rows = pnl_mod.pnl_by_symbol(c, aid_row["id"])
            realized = round(sum(float(r["realized_pnl"] or 0) for r in rows), 2)
    except Exception:
        realized = None

    # --- equity endpoints today ---
    eq = c.execute(
        "SELECT equity FROM equity_curve WHERE ts > ? "
        "AND account_id IN (SELECT id FROM accounts WHERE venue='alpaca_paper') "
        "ORDER BY ts", (day0,)
    ).fetchall()
    equity_start = float(eq[0]["equity"]) if eq else None
    equity_end = float(eq[-1]["equity"]) if eq else None
    equity_chg_pct = (round(100 * (equity_end - equity_start) / equity_start, 2)
                      if equity_start else None)

    summary = (
        f"Close {day0}: {n_fills} fills across {len(traded_syms)} names; "
        f"{accepted} accepted / {rejected} rejected proposals, {exits} exits; "
        f"{throttled} throttled ticks."
    )
    if realized is not None:
        summary += f" Realized P&L ${realized:,.2f}."
    if equity_chg_pct is not None:
        summary += f" Equity {equity_chg_pct:+.2f}%."

    detail = json.dumps({
        "run_counts": run_counts,
        "throttled_ticks": throttled,
        "proposals": proposals, "accepted": accepted, "rejected": rejected,
        "exits": exits, "reject_reasons": reject_reasons,
        "fills": n_fills, "traded_symbols": traded_syms,
        "buy_notional": round(buy_notional, 2), "sell_notional": round(sell_notional, 2),
        "realized_pnl": realized,
        "equity_start": equity_start, "equity_end": equity_end,
        "equity_change_pct": equity_chg_pct,
    }, indent=2)

    return {"summary": summary, "detail": detail, "traded_syms": traded_syms}


def main() -> None:
    review = build_review()
    entry = append_entry(
        kind="session_close",
        summary=review["summary"],
        detail=review["detail"],
        tags=["review", "eod"],
        symbols=review["traded_syms"] or None,
    )
    print("=== End-of-day review ===")
    print(entry["summary"])
    print(entry["detail"])
    print("\n--- Now distil lessons into data/strategy_notes.md ---")
    for q in (
        "Which setups/tiers/catalysts won vs lost today?",
        "Were stops real invalidation or ATR noise?",
        "Did the book over-concentrate in one theme?",
        "Did throttling cost trades — would compact mode earlier have helped?",
        "Did discretionary actions beat or lag the mechanical agent?",
    ):
        print(f"  - {q}")


if __name__ == "__main__":
    main()
