"""Tests for the yfinance-only fallback signal providers.

These must produce well-formed ideas/recommendations without touching the
network, so we patch the module-level `_download` helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.signals.local as local
from src.signals.local import LocalLongTermClient, LocalShortTermClient


def _make_frame(n: int, start: float, drift: float) -> pd.DataFrame:
    """Build a deterministic OHLCV frame with a steady up/down drift."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = np.array([start + drift * i for i in range(n)], dtype=float)
    close = np.clip(close, 1.0, None)
    high = close * 1.01
    low = close * 0.99
    open_ = close
    vol = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


@pytest.fixture
def patched_download(monkeypatch):
    """Patch `_download` to return synthetic frames: uptrend for AAPL, down for KO."""

    def fake_download(symbols, *, period, interval):
        out = {}
        for s in symbols:
            if s == "AAPL":
                out[s] = _make_frame(260, start=100.0, drift=0.5)   # strong uptrend
            elif s == "KO":
                out[s] = _make_frame(260, start=100.0, drift=-0.3)  # downtrend
            else:
                out[s] = _make_frame(260, start=50.0, drift=0.05)   # mild uptrend
        return out

    monkeypatch.setattr(local, "_download", fake_download)


def test_short_term_list_ideas_shape(patched_download):
    client = LocalShortTermClient(universe=["AAPL", "KO", "MSFT"])
    res = client.list_ideas(mode="intraday", tier="C", limit=10)

    assert res["source"] == "local-yfinance"
    assert isinstance(res["ideas"], list)
    syms = [i["symbol"] for i in res["ideas"]]
    # Uptrending names appear; the downtrending one (negative score) is filtered.
    assert "AAPL" in syms
    assert "KO" not in syms
    for idea in res["ideas"]:
        assert idea["entry"] > 0
        assert idea["stop"] < idea["entry"]
        assert idea["direction"] == "long"
        assert idea["tier"] in {"A", "B", "C"}


def test_short_term_ideas_ranked_by_score(patched_download):
    client = LocalShortTermClient(universe=["AAPL", "MSFT"])
    ideas = client.list_ideas(tier="C", limit=10)["ideas"]
    scores = [i["score"] for i in ideas]
    assert scores == sorted(scores, reverse=True) or len({i["tier"] for i in ideas}) > 1


def test_short_term_lookup_ticker(patched_download):
    client = LocalShortTermClient()
    q = client.lookup_ticker("AAPL")
    assert q["symbol"] == "AAPL"
    assert q["price"] > 0
    assert q["last"] == q["price"]


def test_long_term_recommendations_shape(patched_download):
    client = LocalLongTermClient(universe=["AAPL", "KO", "MSFT"])
    res = client.get_recommendations(top_n=5)
    assert res["source"] == "local-yfinance"
    recs = res["recommendations"]
    assert recs, "expected at least one recommendation"
    # Strong uptrend should rank first.
    assert recs[0]["symbol"] == "AAPL"
    for r in recs:
        assert r["price"] > 0
        assert r["rating"] in {"buy", "hold", "avoid"}


def test_long_term_lookup_returns_price(patched_download):
    client = LocalLongTermClient()
    q = client.lookup_ticker("AAPL")
    assert q["price"] > 0
    assert "momentum_6m_pct" in q


def test_long_term_portfolio_suggestion(patched_download):
    client = LocalLongTermClient(universe=["AAPL", "MSFT"])
    sug = client.get_portfolio_suggestion(budget=10_000)
    assert sug["suggestion"]
    total = sum(s["target_weight_pct"] for s in sug["suggestion"])
    assert 99.0 <= total <= 101.0


def test_blocklisted_symbols_never_surface(patched_download):
    client = LocalShortTermClient(universe=["AAPL", "TQQQ", "SQQQ"])
    assert "TQQQ" not in client._universe
    assert "SQQQ" not in client._universe


def test_lifecycle_noops():
    client = LocalShortTermClient()
    client.start()
    assert client.health() is True
    client.stop()
