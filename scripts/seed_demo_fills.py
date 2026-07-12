"""Seed demo filled trades so the P&L analysis has something to show.

The sandbox only fills during market hours (it needs a live yfinance bar), so
after-hours / weekends the book stays empty. This script inserts a small set of
realistic *demo* fills into the `day` and `long` sandbox sub-accounts, with
matching cash-ledger entries and position marks, so the dashboard's P&L tabs
render with real numbers.

All demo rows use agent='demo' so they are easy to identify and remove.

Run:    python -m scripts.seed_demo_fills
Clear:  python -m scripts.seed_demo_fills --clear
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

from src.config import db_path
from src.sandbox import db as dbm

DEMO_AGENT = "demo"


def _clear(conn) -> None:
    # Remove demo orders and their cash-ledger rows / marks.
    order_ids = [r["id"] for r in conn.execute(
        "SELECT id FROM orders WHERE agent=?", (DEMO_AGENT,)).fetchall()]
    conn.execute("DELETE FROM cash_ledger WHERE reason='demo'")
    conn.execute("DELETE FROM positions_snapshot WHERE ts LIKE '%_demo'")
    conn.execute("DELETE FROM orders WHERE agent=?", (DEMO_AGENT,))
    print(f"cleared {len(order_ids)} demo orders")


def _fill(conn, account_id, *, ts, symbol, side, qty, price, fees):
    conn.execute(
        "INSERT INTO orders(account_id, ts, symbol, side, qty, order_type, tif, "
        "status, submitted_at, filled_at, fill_price, fees, agent, venue) "
        "VALUES (?, ?, ?, ?, ?, 'market', 'day', 'filled', ?, ?, ?, ?, ?, 'sandbox')",
        (account_id, ts, symbol, side, qty, ts, ts, price, fees, DEMO_AGENT),
    )
    delta = (-qty * price - fees) if side == "buy" else (qty * price - fees)
    conn.execute(
        "INSERT INTO cash_ledger(account_id, ts, delta, reason) VALUES (?, ?, ?, 'demo')",
        (account_id, ts, delta),
    )


def _mark(conn, account_id, *, day, symbol, qty, avg_cost, mark_price):
    conn.execute(
        "INSERT INTO positions_snapshot(account_id, ts, symbol, qty, avg_cost, mark_price) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (account_id, f"{day}_demo", symbol, qty, avg_cost, mark_price),
    )


def seed(conn) -> None:
    day = dbm.get_account_id(conn, "day")
    long = dbm.get_account_id(conn, "long")
    base = datetime(2026, 7, 6, 14, 0, tzinfo=timezone.utc)

    def t(days: int, mins: int = 0) -> str:
        return (base + timedelta(days=days, minutes=mins)).isoformat()

    # --- Day trader: several intraday round-trips + one open runner ---
    _fill(conn, day, ts=t(0, 0), symbol="NVDA", side="buy", qty=20, price=118.40, fees=0.24)
    _fill(conn, day, ts=t(0, 90), symbol="NVDA", side="sell", qty=20, price=121.10, fees=0.24)
    _fill(conn, day, ts=t(1, 0), symbol="TSLA", side="buy", qty=15, price=250.00, fees=0.38)
    _fill(conn, day, ts=t(1, 120), symbol="TSLA", side="sell", qty=15, price=246.20, fees=0.37)
    _fill(conn, day, ts=t(2, 0), symbol="AMD", side="buy", qty=30, price=140.00, fees=0.42)
    _fill(conn, day, ts=t(2, 60), symbol="AMD", side="sell", qty=30, price=143.75, fees=0.43)
    _fill(conn, day, ts=t(3, 0), symbol="AAPL", side="buy", qty=25, price=210.00, fees=0.52)  # open
    _mark(conn, day, day="2026-07-10", symbol="AAPL", qty=25, avg_cost=210.00, mark_price=214.30)

    # --- Long-term: builds positions, trims one ---
    _fill(conn, long, ts=t(0, 0), symbol="MSFT", side="buy", qty=40, price=430.00, fees=1.72)
    _fill(conn, long, ts=t(1, 0), symbol="GOOGL", side="buy", qty=60, price=175.00, fees=1.05)
    _fill(conn, long, ts=t(2, 0), symbol="JPM", side="buy", qty=50, price=205.00, fees=1.03)
    _fill(conn, long, ts=t(3, 0), symbol="MSFT", side="sell", qty=10, price=448.00, fees=0.45)  # trim
    _mark(conn, long, day="2026-07-10", symbol="MSFT", qty=30, avg_cost=430.00, mark_price=451.20)
    _mark(conn, long, day="2026-07-10", symbol="GOOGL", qty=60, avg_cost=175.00, mark_price=168.90)
    _mark(conn, long, day="2026-07-10", symbol="JPM", qty=50, avg_cost=205.00, mark_price=212.40)
    print("seeded demo fills for day + long sub-accounts")


def main() -> None:
    conn = dbm.get_conn(db_path())
    dbm.migrate(conn)
    dbm.bootstrap_accounts(conn)
    if "--clear" in sys.argv:
        _clear(conn)
    else:
        _clear(conn)  # idempotent: reset any prior demo rows first
        seed(conn)


if __name__ == "__main__":
    main()
