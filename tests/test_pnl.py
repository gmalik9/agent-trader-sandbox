"""Tests for the average-cost P&L analytics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.analysis import pnl as pnl_mod
from src.sandbox import db as dbm


@pytest.fixture
def acct(tmp_db):
    return dbm.get_account_id(tmp_db, "day")


def _order(conn, account_id, *, ts, symbol, side, qty, price, fees=0.0,
           status="filled"):
    conn.execute(
        "INSERT INTO orders(account_id, ts, symbol, side, qty, order_type, tif, "
        "status, submitted_at, filled_at, fill_price, fees, agent, venue) "
        "VALUES (?, ?, ?, ?, ?, 'market', 'day', ?, ?, ?, ?, ?, 'day', 'sandbox')",
        (account_id, ts, symbol, side, qty, status, ts, ts, price, fees),
    )


def _mark(conn, account_id, *, ts, symbol, qty, avg_cost, mark_price):
    conn.execute(
        "INSERT INTO positions_snapshot(account_id, ts, symbol, qty, avg_cost, mark_price) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (account_id, ts, symbol, qty, avg_cost, mark_price),
    )


def test_realized_pnl_full_close(tmp_db, acct):
    t0 = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
    _order(tmp_db, acct, ts=t0.isoformat(), symbol="AAPL", side="buy", qty=10, price=100)
    _order(tmp_db, acct, ts=(t0 + timedelta(hours=1)).isoformat(),
           symbol="AAPL", side="sell", qty=10, price=110)
    rows = pnl_mod.pnl_by_symbol(tmp_db, acct)
    assert len(rows) == 1
    r = rows[0]
    assert r["symbol"] == "AAPL"
    assert r["realized_pnl"] == 100.0        # 10 * (110 - 100)
    assert r["unrealized_pnl"] == 0.0
    assert r["open_qty"] == 0.0
    assert r["net_pnl"] == 100.0


def test_average_cost_on_multiple_buys(tmp_db, acct):
    t0 = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
    _order(tmp_db, acct, ts=t0.isoformat(), symbol="MSFT", side="buy", qty=10, price=100)
    _order(tmp_db, acct, ts=(t0 + timedelta(minutes=1)).isoformat(),
           symbol="MSFT", side="buy", qty=10, price=200)  # avg = 150
    _order(tmp_db, acct, ts=(t0 + timedelta(minutes=2)).isoformat(),
           symbol="MSFT", side="sell", qty=5, price=160)
    rows = pnl_mod.pnl_by_symbol(tmp_db, acct)
    r = rows[0]
    assert r["avg_cost"] == 150.0
    assert r["realized_pnl"] == 50.0         # 5 * (160 - 150)
    assert r["open_qty"] == 15.0


def test_unrealized_uses_latest_mark(tmp_db, acct):
    t0 = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
    _order(tmp_db, acct, ts=t0.isoformat(), symbol="NVDA", side="buy", qty=4, price=100)
    _mark(tmp_db, acct, ts=(t0 + timedelta(hours=2)).isoformat(),
          symbol="NVDA", qty=4, avg_cost=100, mark_price=125)
    rows = pnl_mod.pnl_by_symbol(tmp_db, acct)
    r = rows[0]
    assert r["unrealized_pnl"] == 100.0      # 4 * (125 - 100)
    assert r["mark"] == 125.0
    assert r["net_pnl"] == 100.0
    # Extended fields for the readable trades table.
    assert r["status"] == "Holding"
    assert r["cost_basis"] == 400.0
    assert r["market_value"] == 500.0
    assert r["pnl_pct"] == 25.0              # 100 / 400


def test_status_closed_and_partly_sold(tmp_db, acct):
    t0 = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
    # Fully closed
    _order(tmp_db, acct, ts=t0.isoformat(), symbol="AAPL", side="buy", qty=10, price=100)
    _order(tmp_db, acct, ts=(t0 + timedelta(minutes=1)).isoformat(),
           symbol="AAPL", side="sell", qty=10, price=110)
    # Partly sold (5 of 10)
    _order(tmp_db, acct, ts=(t0 + timedelta(minutes=2)).isoformat(),
           symbol="MSFT", side="buy", qty=10, price=100)
    _order(tmp_db, acct, ts=(t0 + timedelta(minutes=3)).isoformat(),
           symbol="MSFT", side="sell", qty=5, price=120)
    rows = {r["symbol"]: r for r in pnl_mod.pnl_by_symbol(tmp_db, acct)}
    assert rows["AAPL"]["status"] == "Closed"
    assert rows["MSFT"]["status"] == "Holding (partly sold)"


def test_fees_reduce_net(tmp_db, acct):
    t0 = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
    _order(tmp_db, acct, ts=t0.isoformat(), symbol="KO", side="buy", qty=10, price=50, fees=2.5)
    _order(tmp_db, acct, ts=(t0 + timedelta(hours=1)).isoformat(),
           symbol="KO", side="sell", qty=10, price=55, fees=2.5)
    rows = pnl_mod.pnl_by_symbol(tmp_db, acct)
    r = rows[0]
    assert r["realized_pnl"] == 50.0
    assert r["fees"] == 5.0
    assert r["net_pnl"] == 45.0              # 50 - 5


def test_sorted_by_net_descending(tmp_db, acct):
    t0 = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
    _order(tmp_db, acct, ts=t0.isoformat(), symbol="LOSS", side="buy", qty=1, price=100)
    _order(tmp_db, acct, ts=(t0 + timedelta(minutes=1)).isoformat(),
           symbol="LOSS", side="sell", qty=1, price=90)
    _order(tmp_db, acct, ts=(t0 + timedelta(minutes=2)).isoformat(),
           symbol="WIN", side="buy", qty=1, price=100)
    _order(tmp_db, acct, ts=(t0 + timedelta(minutes=3)).isoformat(),
           symbol="WIN", side="sell", qty=1, price=120)
    rows = pnl_mod.pnl_by_symbol(tmp_db, acct)
    assert [r["symbol"] for r in rows] == ["WIN", "LOSS"]


def test_cumulative_timeseries(tmp_db, acct):
    t0 = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
    _order(tmp_db, acct, ts=t0.isoformat(), symbol="AAPL", side="buy", qty=10, price=100)
    _order(tmp_db, acct, ts=(t0 + timedelta(minutes=1)).isoformat(),
           symbol="AAPL", side="sell", qty=5, price=110)   # +50
    _order(tmp_db, acct, ts=(t0 + timedelta(minutes=2)).isoformat(),
           symbol="AAPL", side="sell", qty=5, price=120)   # +100 -> cum 150
    ts = pnl_mod.realized_pnl_timeseries(tmp_db, acct)
    assert [p["cum_realized"] for p in ts] == [0.0, 50.0, 150.0]


def test_totals(tmp_db, acct):
    t0 = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
    _order(tmp_db, acct, ts=t0.isoformat(), symbol="AAPL", side="buy", qty=10, price=100, fees=1)
    _order(tmp_db, acct, ts=(t0 + timedelta(hours=1)).isoformat(),
           symbol="AAPL", side="sell", qty=10, price=110, fees=1)
    rows = pnl_mod.pnl_by_symbol(tmp_db, acct)
    tot = pnl_mod.totals(rows)
    assert tot["realized_pnl"] == 100.0
    assert tot["fees"] == 2.0
    assert tot["net_pnl"] == 98.0


def test_empty_account(tmp_db, acct):
    assert pnl_mod.pnl_by_symbol(tmp_db, acct) == []
    assert pnl_mod.realized_pnl_timeseries(tmp_db, acct) == []
    assert pnl_mod.totals([]) == {"realized_pnl": 0.0, "unrealized_pnl": 0.0,
                                   "fees": 0.0, "net_pnl": 0.0}
