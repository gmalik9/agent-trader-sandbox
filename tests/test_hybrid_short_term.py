"""Tests for the hybrid short-term client (MCP-first, local fallback)."""

from __future__ import annotations

from src.signals.local import HybridShortTermClient, LocalShortTermClient


class FakeReal:
    def __init__(self, ideas_rows=None, raise_on_ideas=False):
        self._rows = ideas_rows
        self._raise = raise_on_ideas
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        pass

    def list_ideas(self, *, mode="intraday", tier="A", limit=10):
        if self._raise:
            raise RuntimeError("scanner down")
        return {"count": len(self._rows or []), "rows": self._rows or []}

    def lookup_ticker(self, ticker, *, interval="5m", period="5d"):
        return {"symbol": ticker, "last": 123.0}

    def get_news(self, ticker, *, days=1, limit=20):
        return {"symbol": ticker, "news": ["real"]}


def test_uses_real_ideas_when_present():
    real = FakeReal(ideas_rows=[{"symbol": "AAPL", "tier": "A"}])
    h = HybridShortTermClient(real, local=LocalShortTermClient(universe=["AAPL"]))
    res = h.list_ideas()
    assert res["rows"] == [{"symbol": "AAPL", "tier": "A"}]


def test_falls_back_to_local_when_real_empty(monkeypatch):
    real = FakeReal(ideas_rows=[])
    local = LocalShortTermClient(universe=["AAPL", "MSFT"])
    # Stub the local provider so we don't hit the network.
    monkeypatch.setattr(local, "list_ideas",
                        lambda **kw: {"count": 1, "rows": [{"symbol": "AAPL"}],
                                       "source": "local-yfinance"})
    h = HybridShortTermClient(real, local=local)
    res = h.list_ideas()
    assert res["source"] == "local-yfinance"
    assert res["rows"] == [{"symbol": "AAPL"}]


def test_falls_back_to_local_when_real_raises(monkeypatch):
    real = FakeReal(raise_on_ideas=True)
    local = LocalShortTermClient(universe=["AAPL"])
    monkeypatch.setattr(local, "list_ideas",
                        lambda **kw: {"count": 1, "rows": [{"symbol": "AAPL"}],
                                       "source": "local-yfinance"})
    h = HybridShortTermClient(real, local=local)
    res = h.list_ideas()
    assert res["source"] == "local-yfinance"


def test_start_survives_real_failure():
    class Boom(FakeReal):
        def start(self):
            raise RuntimeError("cannot start")
    h = HybridShortTermClient(Boom(), local=LocalShortTermClient(universe=["AAPL"]))
    h.start()  # should not raise; real is dropped
    assert h._real is None
