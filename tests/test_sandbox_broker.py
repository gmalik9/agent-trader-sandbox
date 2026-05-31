import pytest

from src.brokers.base import OrderRequest
from src.brokers.sandbox_broker import SandboxBroker
from src.sandbox import db as dbm


def _broker(conn, bars, **kw):
    return SandboxBroker(conn=conn, bar_provider=bars,
                         max_order_usd=kw.get("max_order_usd", 25_000.0),
                         max_symbol_pct=kw.get("max_symbol_pct", 25.0))


def test_buy_then_sell_round_trip_conserves_cash(tmp_db, stub_bars, monkeypatch):
    monkeypatch.setenv("SLIPPAGE_BPS", "0")
    monkeypatch.setenv("COMMISSION_BPS", "0")
    from src import config
    config.get_settings.cache_clear()

    stub_bars.set("AAPL", o=100, h=101, l=99, c=100)
    b = _broker(tmp_db, stub_bars)
    start = b.get_account("day").cash

    buy = b.place_order(OrderRequest("AAPL", "buy", 10, sub_account="day", agent="test"))
    assert buy.status == "filled"
    after_buy = b.get_account("day")
    assert after_buy.cash < start
    assert after_buy.cash + after_buy.positions_value == pytest.approx(
        start - buy.fees, rel=0, abs=1e-6,
    )

    sell = b.place_order(OrderRequest("AAPL", "sell", 10, sub_account="day", agent="test"))
    assert sell.status == "filled"
    after = b.get_account("day")
    assert after.cash == pytest.approx(start - buy.fees - sell.fees, rel=0, abs=1e-6)


def test_short_sale_rejected(tmp_db, stub_bars):
    stub_bars.set("AAPL", o=100, h=101, l=99, c=100)
    b = _broker(tmp_db, stub_bars)
    res = b.place_order(OrderRequest("AAPL", "sell", 5, sub_account="day", agent="test"))
    assert res.status == "rejected"


def test_blocklist_rejects(tmp_db, stub_bars):
    stub_bars.set("TQQQ", o=50, h=51, l=49, c=50)
    b = _broker(tmp_db, stub_bars)
    res = b.place_order(OrderRequest("TQQQ", "buy", 10, sub_account="day", agent="test"))
    assert res.status == "rejected"


def test_per_order_cap_rejects(tmp_db, stub_bars):
    stub_bars.set("AAPL", o=100, h=101, l=99, c=100)
    b = _broker(tmp_db, stub_bars, max_order_usd=500.0)
    res = b.place_order(OrderRequest("AAPL", "buy", 10, sub_account="day", agent="test"))
    assert res.status == "rejected"


def test_per_symbol_pct_rejects(tmp_db, stub_bars):
    stub_bars.set("AAPL", o=100, h=101, l=99, c=100)
    b = _broker(tmp_db, stub_bars, max_symbol_pct=1.0)  # day account = $30k, 1% = $300
    res = b.place_order(OrderRequest("AAPL", "buy", 10, sub_account="day", agent="test"))
    assert res.status == "rejected"


def test_insufficient_cash_rejects(tmp_db, stub_bars):
    stub_bars.set("AAPL", o=100, h=101, l=99, c=100)
    b = _broker(tmp_db, stub_bars)
    res = b.place_order(OrderRequest("AAPL", "buy", 1_000_000, sub_account="day", agent="test"))
    assert res.status == "rejected"
