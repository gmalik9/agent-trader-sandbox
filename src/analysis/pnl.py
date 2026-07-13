"""P&L analytics — pure functions over the SQLite ledger.

Kept free of Streamlit so the accounting can be unit-tested. The dashboard
(`app.py`) wraps these in DataFrames for display.

Accounting model: average-cost. A sell realizes P&L of
``qty * (sell_price - avg_cost)`` against the running average cost of the open
long position. Fees are tracked separately and netted into the totals. Shorting
is not supported by the sandbox, so sells beyond the open quantity simply drive
the position to zero.
"""

from __future__ import annotations

import sqlite3
from typing import Any


def latest_marks(conn: sqlite3.Connection, account_id: int) -> dict[str, float]:
    """Most recent mark price per symbol from position snapshots."""
    rows = conn.execute(
        "SELECT symbol, mark_price FROM positions_snapshot ps "
        "WHERE account_id=? AND ts=(SELECT MAX(ts) FROM positions_snapshot "
        "WHERE account_id=ps.account_id AND symbol=ps.symbol)",
        (account_id,),
    ).fetchall()
    return {r["symbol"]: float(r["mark_price"]) for r in rows}


def _filled_orders(conn: sqlite3.Connection, account_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT ts, symbol, side, qty, fill_price, COALESCE(fees,0) AS fees "
        "FROM orders WHERE account_id=? AND status='filled' AND fill_price IS NOT NULL "
        "ORDER BY ts ASC",
        (account_id,),
    ).fetchall()


def pnl_by_symbol(conn: sqlite3.Connection, account_id: int,
                  marks: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Per-symbol realized/unrealized/fees/net P&L, sorted by net descending."""
    if marks is None:
        marks = latest_marks(conn, account_id)
def _apply_fill(s: dict[str, float], side: str, qty: float, price: float) -> None:
    """Apply one fill to a signed position state, accruing realized P&L.

    Supports long and short positions. `s['qty']` is signed (negative = short),
    `s['avg_cost']` is the entry price of the currently open side, and
    `s['realized']` accumulates locked-in P&L (long: sell above cost; short: buy
    below cost).
    """
    signed = qty if side == "buy" else -qty
    pos = s["qty"]
    if pos == 0 or (pos > 0) == (signed > 0):
        # Opening or adding in the same direction → weighted-average cost.
        new_qty = pos + signed
        if new_qty != 0:
            s["avg_cost"] = (s["avg_cost"] * pos + price * signed) / new_qty
        s["qty"] = new_qty
        return
    # Opposite direction → reduce / close / flip and realize.
    reduce_qty = min(abs(signed), abs(pos))
    direction = 1.0 if pos > 0 else -1.0  # long realizes (price-cost); short the inverse
    s["realized"] += reduce_qty * (price - s["avg_cost"]) * direction
    new_qty = pos + signed
    if abs(new_qty) <= 1e-9:
        s["qty"] = 0.0
        s["avg_cost"] = 0.0
    elif (new_qty > 0) != (pos > 0):
        # Flipped through zero → the remainder opens a new position at this price.
        s["qty"] = new_qty
        s["avg_cost"] = price
    else:
        s["qty"] = new_qty  # partial reduce, basis unchanged


def pnl_by_symbol(conn: sqlite3.Connection, account_id: int,
                  marks: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Per-symbol realized/unrealized/fees/net P&L, sorted by net descending."""
    if marks is None:
        marks = latest_marks(conn, account_id)
    state: dict[str, dict[str, float]] = {}
    for r in _filled_orders(conn, account_id):
        sym = r["symbol"]
        s = state.setdefault(sym, {"qty": 0.0, "avg_cost": 0.0, "realized": 0.0,
                                    "fees": 0.0, "trades": 0.0})
        s["fees"] += float(r["fees"])
        s["trades"] += 1
        _apply_fill(s, r["side"], float(r["qty"]), float(r["fill_price"]))

    out: list[dict[str, Any]] = []
    for sym, s in state.items():
        mark = marks.get(sym, s["avg_cost"])
        open_qty = s["qty"]
        unrealized = open_qty * (mark - s["avg_cost"]) if abs(open_qty) > 1e-9 else 0.0
        net = s["realized"] + unrealized - s["fees"]
        cost_basis = abs(open_qty) * s["avg_cost"] if abs(open_qty) > 1e-9 else 0.0
        market_value = open_qty * mark if abs(open_qty) > 1e-9 else 0.0
        # Human-readable position status.
        if open_qty > 1e-9:
            status = "Long" if s["realized"] == 0.0 else "Long (partly closed)"
        elif open_qty < -1e-9:
            status = "Short" if s["realized"] == 0.0 else "Short (partly covered)"
        else:
            status = "Closed"
        # P&L % relative to invested/committed cost basis.
        if abs(open_qty) > 1e-9 and cost_basis > 0:
            pnl_pct = 100.0 * unrealized / cost_basis
        else:
            pnl_pct = 0.0
        out.append({
            "symbol": sym,
            "status": status,
            "realized_pnl": round(s["realized"], 2),
            "unrealized_pnl": round(unrealized, 2),
            "fees": round(s["fees"], 2),
            "net_pnl": round(net, 2),
            "pnl_pct": round(pnl_pct, 2),
            "open_qty": round(open_qty, 4),
            "avg_cost": round(s["avg_cost"], 2) if abs(open_qty) > 1e-9 else 0.0,
            "mark": round(mark, 2) if abs(open_qty) > 1e-9 else 0.0,
            "cost_basis": round(cost_basis, 2),
            "market_value": round(market_value, 2),
            "trades": int(s["trades"]),
        })
    out.sort(key=lambda d: d["net_pnl"], reverse=True)
    return out


def realized_pnl_timeseries(conn: sqlite3.Connection,
                            account_id: int) -> list[dict[str, Any]]:
    """Cumulative realized P&L (net of fees) after each fill, in time order."""
    state: dict[str, dict[str, float]] = {}
    cum = 0.0
    points: list[dict[str, Any]] = []
    for r in _filled_orders(conn, account_id):
        sym = r["symbol"]
        s = state.setdefault(sym, {"qty": 0.0, "avg_cost": 0.0, "realized": 0.0})
        cum -= float(r["fees"])
        before = s["realized"]
        _apply_fill(s, r["side"], float(r["qty"]), float(r["fill_price"]))
        cum += s["realized"] - before
        points.append({"ts": r["ts"], "cum_realized": round(cum, 2)})
    return points


def totals(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-symbol rows into account-level totals."""
    return {
        "realized_pnl": round(sum(r["realized_pnl"] for r in rows), 2),
        "unrealized_pnl": round(sum(r["unrealized_pnl"] for r in rows), 2),
        "fees": round(sum(r["fees"] for r in rows), 2),
        "net_pnl": round(sum(r["net_pnl"] for r in rows), 2),
    }
