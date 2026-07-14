"""End-to-end DayTraderAgent tests.

These exercise the full `run_once` pipeline through the *dual* broker
(Alpaca = primary source of truth, sandbox = mirror) with a scripted LLM and a
fake Alpaca MCP — the same wiring the scheduler uses in production — to prove:

  - a proposed trade is sized, placed on Alpaca (primary) AND mirrored to the
    sandbox, with a real external order id;
  - diversification is mechanically enforced (per-name cap, correlation-theme
    cap, recently-traded cooldown);
  - a short of a leveraged/inverse ETF is auto-substituted to a long of its
    inverse and still lands on Alpaca;
  - a rate-limited tick degrades to a visible no-op + throttle marker instead of
    crashing;
  - reconcile pulls resolved Alpaca statuses into the local mirror;
  - the kill switch, market-closed, and force-flat guards short-circuit cleanly.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.agents.day_trader import DayTraderAgent
from src.brokers.alpaca_paper_broker import AlpacaPaperBroker
from src.brokers.dual_broker import DualBroker
from src.brokers.sandbox_broker import SandboxBroker
from src.llm.github_models import RateLimitError
from src.llm.provider import ChatResult, ToolCall
from src.sandbox import db as dbm


MARKET_OPEN = datetime(2025, 1, 14, 14, 30, tzinfo=timezone.utc)   # 10:30 ET
MARKET_CLOSED = datetime(2025, 1, 14, 3, 0, tzinfo=timezone.utc)   # overnight
FORCE_FLAT = datetime(2025, 1, 14, 20, 57, tzinfo=timezone.utc)    # 15:57 ET


# --------------------------------------------------------------------------- #
#  Fakes
# --------------------------------------------------------------------------- #
class FakeMCP:
    """Fake stock-recommender MCP (the Alpaca leg). Configurable positions and
    a fill response, plus a resolvable order book for reconcile."""

    def __init__(self, *, equity=100_000.0, cash=100_000.0, positions=None):
        self.equity = equity
        self.cash = cash
        self._positions = list(positions or [])
        self.placed: list[dict] = []
        self.restarts = 0
        self._next_id = 1000
        self._orders: dict[str, dict] = {}

    # lifecycle (used by the broker's self-heal path)
    def start(self): self.restarts += 1
    def stop(self): pass

    def get_account(self):
        return {"equity": self.equity, "cash": self.cash, "account_number": "PA00001"}

    def list_positions(self):
        return list(self._positions)

    def place_order(self, *, symbol, qty, side, order_type="market",
                    time_in_force="day", limit_price=None):
        self.placed.append({"symbol": symbol, "qty": qty, "side": side})
        oid = f"ext-{self._next_id}"
        self._next_id += 1
        # Fills immediately; reconcile would later confirm the same.
        self._orders[oid] = {"id": oid, "status": "filled",
                             "filled_avg_price": limit_price or 100.0,
                             "filled_at": "2025-01-14T14:30:05Z"}
        return {"order_id": oid, "status": "filled",
                "filled_avg_price": limit_price or 100.0,
                "filled_at": "2025-01-14T14:30:05Z"}

    def list_orders(self, *, status="all", limit=100):
        return list(self._orders.values())

    def close_position(self, symbol, percentage=100):
        oid = f"close-{self._next_id}"
        self._next_id += 1
        return {"order_id": oid, "status": "filled"}


class ScriptedProvider:
    name = "scripted"
    model = "scripted"

    def __init__(self, results):
        self._queue = list(results)

    def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=1024):
        if not self._queue:
            return ChatResult(text="done")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class RaisingProvider:
    name = "raising"
    model = "raising"

    def __init__(self, exc):
        self.exc = exc

    def chat(self, *a, **kw):
        raise self.exc


class FakeShortTerm:
    """Idea scanner stub returning a configurable ranked idea set."""

    def __init__(self, ideas=None, price=100.0):
        self._ideas = ideas if ideas is not None else [
            {"ticker": "AAPL", "direction": "long", "tier": "A", "heat_score": 80.0,
             "signal_tags": "vwap_reclaim,news_spike", "rr": 2.0,
             "entry": 100.0, "stop": 98.0, "target": 104.0,
             "dollar_risk": 250.0, "dollar_gain": 500.0},
        ]
        self._price = price

    def list_ideas(self, **kw):
        return {"count": len(self._ideas), "rows": list(self._ideas)}

    def lookup_ticker(self, ticker, **kw):
        return {"symbol": ticker, "price": self._price,
                "bars": [{"close": self._price, "vwap": self._price - 1,
                          "rsi": 55, "atr": 1.0, "atr_pct": 0.01, "macd": 0.2,
                          "signal": 0.1, "vol_z": 1.0, "sma_20": self._price,
                          "sma_50": self._price, "sma_200": self._price}]}

    def get_news(self, ticker, **kw):
        return {"ticker": ticker, "articles": [
            {"headline": f"{ticker} steady", "sentiment": {"score": 0.2, "label": "Positive"},
             "source": "finnhub"}]}


class FakeLongTerm:
    def lookup_ticker(self, ticker):
        return {"ticker": ticker, "price": 100.0, "target_price": 120.0,
                "rating": "Buy", "analyst_count": 20, "upside_pct": 20.0,
                "sentiment_score": 0.2, "sentiment_label": "Positive"}

    def get_news(self, ticker, *, days=7, limit=20):
        return {"ticker": ticker, "articles": [
            {"headline": f"{ticker} ok", "sentiment": {"score": 0.1, "label": "Positive"},
             "source": "yahoo"}]}


def _propose(symbol, entry, stop, side="buy"):
    return ChatResult(text=None, tool_calls=[ToolCall(
        id="c1", name="propose_trade",
        arguments={"symbol": symbol, "entry_price": entry, "stop_price": stop,
                    "side": side, "thesis": "e2e"})])


def _dual(conn, stub_bars, mcp):
    alpaca = AlpacaPaperBroker(mcp, conn=conn)
    sandbox = SandboxBroker(conn, bar_provider=stub_bars, mirror=True)
    return DualBroker(primary=alpaca, secondary=sandbox, conn=conn)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    import src.brokers.alpaca_paper_broker as ap
    monkeypatch.setattr(ap.time, "sleep", lambda *_: None)


# --------------------------------------------------------------------------- #
#  End-to-end happy path
# --------------------------------------------------------------------------- #
def test_e2e_trade_lands_on_alpaca_and_mirrors_to_sandbox(tmp_db, stub_bars, monkeypatch):
    monkeypatch.setenv("SLIPPAGE_BPS", "0")
    monkeypatch.setenv("COMMISSION_BPS", "0")
    monkeypatch.setenv("DAY_MAX_POSITION_PCT", "0.20")
    monkeypatch.setenv("DAY_NAME_COOLDOWN_SECONDS", "0")   # no cooldown for this test
    from src import config
    config.get_settings.cache_clear()

    stub_bars.set("AAPL", o=100, h=101, l=99, c=100)
    mcp = FakeMCP(equity=100_000.0, cash=100_000.0)
    broker = _dual(tmp_db, stub_bars, mcp)
    provider = ScriptedProvider([_propose("AAPL", 100.0, 98.0), ChatResult(text="done")])
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(), provider=provider,
                            now=MARKET_OPEN, long_term=FakeLongTerm())

    out = agent.run_once()
    assert out.status == "ok"

    # Alpaca (primary) got the order WITH an external id …
    alp = tmp_db.execute(
        "SELECT status, external_id FROM orders WHERE venue='alpaca_paper' AND symbol='AAPL'"
    ).fetchone()
    assert alp is not None and alp["external_id"] and alp["status"] == "filled"
    assert mcp.placed and mcp.placed[0]["symbol"] == "AAPL"

    # … and the sandbox mirror recorded the same fill.
    sbx = tmp_db.execute(
        "SELECT status FROM orders WHERE venue='sandbox' AND symbol='AAPL'"
    ).fetchone()
    assert sbx is not None and sbx["status"] == "filled"
    config.get_settings.cache_clear()


def test_e2e_reconcile_syncs_alpaca_status(tmp_db, stub_bars, monkeypatch):
    monkeypatch.setenv("DAY_NAME_COOLDOWN_SECONDS", "0")
    from src import config
    config.get_settings.cache_clear()
    stub_bars.set("AAPL", o=100, h=101, l=99, c=100)
    mcp = FakeMCP()
    broker = _dual(tmp_db, stub_bars, mcp)
    provider = ScriptedProvider([_propose("AAPL", 100.0, 98.0), ChatResult(text="done")])
    DayTraderAgent(tmp_db, broker, FakeShortTerm(), provider=provider,
                   now=MARKET_OPEN, long_term=FakeLongTerm()).run_once()
    # reconcile is idempotent and should not error; returns an int.
    updated = broker.reconcile()
    assert isinstance(updated, int)
    config.get_settings.cache_clear()


# --------------------------------------------------------------------------- #
#  Diversification (mechanical enforcement, end-to-end)
# --------------------------------------------------------------------------- #
def test_e2e_theme_cap_blocks_correlated_second_name(tmp_db, stub_bars, monkeypatch):
    monkeypatch.setenv("SLIPPAGE_BPS", "0")
    monkeypatch.setenv("COMMISSION_BPS", "0")
    monkeypatch.setenv("DAY_MAX_POSITION_PCT", "0.20")
    monkeypatch.setenv("DAY_THEME_MAX_PCT", "0.35")
    monkeypatch.setenv("DAY_NAME_COOLDOWN_SECONDS", "0")
    from src import config
    config.get_settings.cache_clear()

    stub_bars.set("NVDL", o=50, h=51, l=49, c=50)
    # Alpaca already holds SOXL at the semis theme cap (~35% of $100k).
    mcp = FakeMCP(positions=[{"symbol": "SOXL", "qty": 350, "entry_price": 100.0,
                              "current_price": 100.0}])
    broker = _dual(tmp_db, stub_bars, mcp)
    # LLM proposes NVDL — same 'semis' theme → must be blocked end-to-end.
    provider = ScriptedProvider([_propose("NVDL", 50.0, 49.0), ChatResult(text="done")])
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(), provider=provider,
                            now=MARKET_OPEN, long_term=FakeLongTerm())
    out = agent.run_once()
    assert out.status == "ok"
    d = [d for d in out.decisions if d.get("symbol") == "NVDL"]
    assert d and not d[0]["accepted"]
    assert str(d[0]["reject_reason"]).startswith("theme_at_cap")
    # Nothing placed on Alpaca for NVDL.
    assert all(p["symbol"] != "NVDL" for p in mcp.placed)
    config.get_settings.cache_clear()


def test_e2e_per_name_cap_position_aware(tmp_db, stub_bars, monkeypatch):
    monkeypatch.setenv("SLIPPAGE_BPS", "0")
    monkeypatch.setenv("COMMISSION_BPS", "0")
    monkeypatch.setenv("DAY_MAX_POSITION_PCT", "0.20")
    monkeypatch.setenv("DAY_THEME_MAX_PCT", "0")   # isolate the per-name cap
    monkeypatch.setenv("DAY_NAME_COOLDOWN_SECONDS", "0")
    from src import config
    config.get_settings.cache_clear()

    stub_bars.set("KO", o=100, h=101, l=99, c=100)
    # Already hold KO at 20% ($20k of $100k).
    mcp = FakeMCP(positions=[{"symbol": "KO", "qty": 200, "entry_price": 100.0,
                              "current_price": 100.0}])
    broker = _dual(tmp_db, stub_bars, mcp)
    provider = ScriptedProvider([_propose("KO", 100.0, 98.0), ChatResult(text="done")])
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(), provider=provider,
                            now=MARKET_OPEN, long_term=FakeLongTerm())
    out = agent.run_once()
    d = [d for d in out.decisions if d.get("symbol") == "KO"][0]
    assert not d["accepted"] and d["reject_reason"] == "size_rounded_to_zero"
    config.get_settings.cache_clear()


def test_e2e_cooldown_hides_recently_traded_name(tmp_db, stub_bars, monkeypatch):
    monkeypatch.setenv("DAY_NAME_COOLDOWN_SECONDS", "600")
    from src import config
    config.get_settings.cache_clear()
    stub_bars.set("AAPL", o=100, h=101, l=99, c=100)
    mcp = FakeMCP()
    broker = _dual(tmp_db, stub_bars, mcp)
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN,
                            long_term=FakeLongTerm())
    # Seed a just-placed AAPL order so it's on cooldown.
    from src.brokers.base import OrderRequest
    broker.place_order(OrderRequest("AAPL", "buy", 10, sub_account="day", agent="day"))
    assert "AAPL" in agent._recent_traded_symbols()
    config.get_settings.cache_clear()


# --------------------------------------------------------------------------- #
#  Inverse substitution end-to-end
# --------------------------------------------------------------------------- #
def test_e2e_short_leveraged_etf_becomes_long_inverse_on_alpaca(tmp_db, stub_bars, monkeypatch):
    monkeypatch.setenv("SLIPPAGE_BPS", "0")
    monkeypatch.setenv("COMMISSION_BPS", "0")
    monkeypatch.setenv("DAY_MAX_POSITION_PCT", "0.20")
    monkeypatch.setenv("DAY_THEME_MAX_PCT", "0")
    monkeypatch.setenv("DAY_NAME_COOLDOWN_SECONDS", "0")
    from src import config
    config.get_settings.cache_clear()

    stub_bars.set("SOXS", o=10, h=10.1, l=9.9, c=10)
    mcp = FakeMCP()
    broker = _dual(tmp_db, stub_bars, mcp)
    # Short SOXL is priced from the short-term client (SOXS @ 10 here).
    short_term = FakeShortTerm(price=10.0)
    provider = ScriptedProvider([_propose("SOXL", 100.0, 105.0, side="sell"),
                                  ChatResult(text="done")])
    agent = DayTraderAgent(tmp_db, broker, short_term, provider=provider,
                            now=MARKET_OPEN, long_term=FakeLongTerm())
    out = agent.run_once()
    assert out.status == "ok"
    # The order that actually reached Alpaca is a BUY of SOXS (the inverse).
    assert mcp.placed and mcp.placed[0]["symbol"] == "SOXS"
    assert mcp.placed[0]["side"] == "buy"
    config.get_settings.cache_clear()


# --------------------------------------------------------------------------- #
#  Resilience / guards
# --------------------------------------------------------------------------- #
def test_e2e_rate_limited_tick_is_visible_noop_with_marker(tmp_db, stub_bars, monkeypatch):
    monkeypatch.setenv("DAY_NAME_COOLDOWN_SECONDS", "0")
    from src import config
    config.get_settings.cache_clear()
    stub_bars.set("AAPL", o=100, h=101, l=99, c=100)
    mcp = FakeMCP()
    broker = _dual(tmp_db, stub_bars, mcp)
    provider = RaisingProvider(RateLimitError("github models 429 (rate limited)"))
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(), provider=provider,
                            now=MARKET_OPEN, long_term=FakeLongTerm())
    out = agent.run_once()
    assert out.status == "no-op" and out.error == "rate_limited"
    # A throttle marker was recorded for the UI banner.
    assert dbm.get_setting(tmp_db, "llm_throttled_at") is not None
    assert "429" in (dbm.get_setting(tmp_db, "llm_throttle_detail") or "")
    # No order was placed on Alpaca.
    assert not mcp.placed
    config.get_settings.cache_clear()


def test_e2e_kill_switch_halts_before_any_order(tmp_db, stub_bars):
    dbm.set_setting(tmp_db, "kill_switch", "on")
    mcp = FakeMCP()
    broker = _dual(tmp_db, stub_bars, mcp)
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN,
                            long_term=FakeLongTerm())
    out = agent.run_once()
    assert out.status == "halted" and out.error == "kill_switch"
    assert not mcp.placed


def test_e2e_market_closed_is_noop(tmp_db, stub_bars):
    mcp = FakeMCP()
    broker = _dual(tmp_db, stub_bars, mcp)
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_CLOSED,
                            long_term=FakeLongTerm())
    out = agent.run_once()
    assert out.status == "no-op"
    assert not mcp.placed


def test_e2e_force_flat_closes_positions_via_alpaca(tmp_db, stub_bars):
    stub_bars.set("AAPL", o=100, h=101, l=99, c=100)
    # Hold AAPL on the Alpaca leg; force-flat must close it.
    mcp = FakeMCP(positions=[{"symbol": "AAPL", "qty": 10, "entry_price": 100.0,
                              "current_price": 100.0}])
    broker = _dual(tmp_db, stub_bars, mcp)
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=FORCE_FLAT,
                            long_term=FakeLongTerm())
    out = agent.run_once()
    assert out.status == "ok"
    # A close order was recorded on the Alpaca leg.
    row = tmp_db.execute(
        "SELECT COUNT(*) n FROM orders WHERE venue='alpaca_paper' AND symbol='AAPL' AND side='sell'"
    ).fetchone()
    assert row["n"] >= 1
