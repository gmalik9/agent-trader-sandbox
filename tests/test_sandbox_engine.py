from datetime import datetime, timezone

import pytest

from src.sandbox.engine import Bar, OrderSpec, simulate_fill


def _bar(o=100.0, h=105.0, l=95.0, c=102.0):
    return Bar(ts=datetime.now(timezone.utc), open=o, high=h, low=l, close=c, volume=1e6)


def test_market_buy_applies_positive_slippage():
    res = simulate_fill(OrderSpec("AAPL", "buy", 10, "market"), _bar(c=100.0),
                        slippage_bps=10, commission_bps=0)
    assert res.filled is True
    assert res.fill_price == pytest.approx(100.0 * 1.001, rel=1e-9)


def test_market_sell_applies_negative_slippage():
    res = simulate_fill(OrderSpec("AAPL", "sell", 10, "market"), _bar(c=100.0),
                        slippage_bps=10, commission_bps=0)
    assert res.fill_price == pytest.approx(100.0 * 0.999, rel=1e-9)


def test_commission_is_applied():
    res = simulate_fill(OrderSpec("AAPL", "buy", 10, "market"), _bar(c=100.0),
                        slippage_bps=0, commission_bps=5)
    notional = 100.0 * 10
    assert res.fees == pytest.approx(notional * 5 / 10_000)


def test_limit_fills_inside_band():
    res = simulate_fill(OrderSpec("AAPL", "buy", 5, "limit", limit_price=99.0),
                        _bar(l=95.0, h=105.0), slippage_bps=0, commission_bps=0)
    assert res.filled and res.fill_price == 99.0


def test_limit_skips_outside_band():
    res = simulate_fill(OrderSpec("AAPL", "buy", 5, "limit", limit_price=90.0),
                        _bar(l=95.0, h=105.0))
    assert res.filled is False and res.reason == "limit_not_touched"


def test_no_bar_means_no_fill():
    res = simulate_fill(OrderSpec("AAPL", "buy", 5, "market"), None)
    assert res.filled is False and res.reason == "no_bar"
