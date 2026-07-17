#!/usr/bin/env python3
"""Session snapshot — a fast, READ-ONLY health + state report for the day trader.

Run inside the scheduler container so the paths and (optional) Alpaca client are
available:

    docker compose exec -T scheduler python \
        .github/skills/autonomous-day-trader/scripts/session_snapshot.py

It never writes to the database (opens SQLite read-only), so it is safe to run
anytime alongside the live scheduler.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

# Running a script by path puts the script's own dir on sys.path, not the app
# root — add the repo root (4 levels up) so `import src.*` works.
_APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

try:
    from src.config import db_path
    DB = str(db_path())
except Exception:  # pragma: no cover - fallback if run outside the app
    DB = "data/sandbox.sqlite"

try:
    from src.sandbox.clock import is_market_open as _is_market_open
except Exception:  # pragma: no cover
    _is_market_open = None


def _ro_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _hdr(t: str) -> None:
    print(f"\n=== {t} ===")


def main() -> None:
    now = datetime.now(timezone.utc)
    print(f"snapshot @ {now.isoformat(timespec='seconds')} (UTC)")

    market_open = _is_market_open(now) if _is_market_open else None
    print(f"market_open: {market_open}")

    c = _ro_conn()

    _hdr("scheduler / toggles")
    for k in ("scheduler_heartbeat", "kill_switch", "compact_prompt",
              "llm_throttled_at", "llm_throttle_detail"):
        row = c.execute("SELECT value FROM settings WHERE key=?", (k,)).fetchone()
        val = row["value"] if row else "(unset)"
        if k == "llm_throttle_detail" and row:
            val = val[:80]
        print(f"  {k}: {val}")

    _hdr("recent day runs")
    rows = c.execute(
        "SELECT id, ts, status, latency_ms, error FROM agent_runs "
        "WHERE agent='day' ORDER BY id DESC LIMIT 8"
    ).fetchall()
    for r in rows:
        lat = round((r["latency_ms"] or 0) / 1000, 1)
        err = (r["error"] or "")[:34]
        print(f"  #{r['id']} {r['ts'][11:19]} {r['status']:6} {lat:>5}s {err}")

    _hdr("today's decisions (day)")
    day0 = now.strftime("%Y-%m-%d")
    rows = c.execute(
        "SELECT id, ts, status, decisions FROM agent_runs "
        "WHERE agent='day' AND ts > ? ORDER BY id DESC LIMIT 12",
        (day0,),
    ).fetchall()
    n_props = n_acc = n_exit = 0
    for r in rows:
        try:
            ds = json.loads(r["decisions"] or "[]")
        except Exception:
            ds = []
        for d in ds:
            if not isinstance(d, dict):
                continue
            if "agent_exit" in d or "stop_exit" in d:
                n_exit += 1
            elif "symbol" in d:
                n_props += 1
                if d.get("accepted"):
                    n_acc += 1
    print(f"  runs today: {len(rows)} | proposals: {n_props} "
          f"(accepted {n_acc}) | exits: {n_exit}")
    # last few concrete decisions
    for r in rows[:4]:
        try:
            ds = json.loads(r["decisions"] or "[]")
        except Exception:
            ds = []
        parts = []
        for d in ds:
            if not isinstance(d, dict):
                continue
            if "symbol" in d:
                parts.append(f"{d.get('side','?')} {d.get('symbol')}"
                             f"{'' if d.get('accepted') else ' [rej:'+str(d.get('reject_reason'))+']'}")
            elif "agent_exit" in d:
                parts.append(f"exit {d['agent_exit'].get('symbol')}")
            elif "stop_exit" in d:
                parts.append(f"stop {d['stop_exit'].get('symbol')}")
        if parts:
            print(f"    #{r['id']} {r['ts'][11:19]}: {', '.join(parts)}")

    _hdr("active stop plans")
    rows = c.execute(
        "SELECT symbol, side, entry_price, stop_price, target_price FROM position_plans "
        "WHERE active=1 ORDER BY symbol"
    ).fetchall()
    if not rows:
        print("  (none active)")
    for r in rows:
        print(f"  {r['symbol']:20} {r['side']:4} entry={r['entry_price']} "
              f"stop={r['stop_price']} target={r['target_price']}")

    _hdr("open-holdings cost (sandbox mirror, per symbol)")
    rows = c.execute(
        "SELECT symbol, "
        "SUM(CASE WHEN side='buy' THEN qty ELSE -qty END) AS net_qty, "
        "SUM(CASE WHEN side='buy' THEN qty*COALESCE(fill_price,0) "
        "         ELSE -qty*COALESCE(fill_price,0) END)+SUM(COALESCE(fees,0)) AS net_cost "
        "FROM orders WHERE status='filled' AND venue='sandbox' GROUP BY symbol"
    ).fetchall()
    tot = 0.0
    for r in rows:
        if round(r["net_qty"] or 0, 4) == 0.0:
            continue
        tot += r["net_cost"] or 0.0
        print(f"  {r['symbol']:20} net_qty={r['net_qty']:>8.0f} cost=${r['net_cost']:>10,.0f}")
    print(f"  TOTAL open cost = ${tot:,.0f}")

    # Optional: live Alpaca account (best-effort; needs the MCP env).
    _hdr("live Alpaca account (best-effort)")
    try:
        from src.mcp_clients.long_term import LongTermClient
        lt = LongTermClient()
        lt.start()
        try:
            a = lt.get_account() or {}
            eq = float(a.get("equity", 0) or 0)
            print(f"  equity=${eq:,.0f} cash=${float(a.get('cash',0) or 0):,.0f} "
                  f"buying_power=${float(a.get('buying_power',0) or 0):,.0f}")
            gross = 0.0
            for p in lt.list_positions():
                q = float(p.get("qty", 0) or 0)
                px = float(p.get("current_price") or 0)
                mv = q * px
                gross += abs(mv)
                side = "LONG" if q > 0 else "SHORT"
                pct = (100 * abs(mv) / eq) if eq else 0
                print(f"  {p['symbol']:20} {side:5} {q:>7.0f} = ${mv:>11,.0f} "
                      f"({pct:4.1f}%) uPnL=${float(p.get('unrealized_pl') or 0):>7.0f}")
            if eq:
                print(f"  gross exposure = ${gross:,.0f} ({100*gross/eq:.0f}% of equity)")
        finally:
            lt.stop()
    except Exception as e:  # noqa: BLE001
        print(f"  (Alpaca client unavailable: {type(e).__name__})")


if __name__ == "__main__":
    main()
