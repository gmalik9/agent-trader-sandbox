"""Day-Trader agent tests with scripted LLM and fake ShortTerm client."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.agents.day_trader import DayTraderAgent
from src.brokers.sandbox_broker import SandboxBroker
from src.llm.provider import ChatResult, ToolCall
from src.sandbox import db as dbm


# A wall clock inside regular market hours (Tuesday 14:30 UTC = 10:30 ET, market open).
MARKET_OPEN = datetime(2025, 1, 14, 14, 30, tzinfo=timezone.utc)
FORCE_FLAT = datetime(2025, 1, 14, 20, 57, tzinfo=timezone.utc)  # 15:57 ET


class FakeShortTerm:
    """Drop-in stand-in for ShortTermClient — only methods the agent calls."""

    def __init__(self) -> None:
        self.calls = []

    def list_ideas(self, **kw):
        self.calls.append(("list_ideas", kw))
        return {"ideas": [{"symbol": "AAPL", "tier": "A"}]}

    def lookup_ticker(self, ticker, **kw):
        self.calls.append(("lookup_ticker", ticker, kw))
        return {"symbol": ticker, "last": 150.0}

    def get_news(self, ticker, **kw):
        self.calls.append(("get_news", ticker, kw))
        return {"symbol": ticker, "news": [], "sentiment": "neutral"}


def test_summarize_quote_derives_indicator_signals():
    from src.agents.day_trader import _summarize_quote
    raw = {"ticker": "AAPL", "bars": [{
        "close": 320.0, "vwap": 316.0, "rsi": 72.0, "atr": 1.5, "atr_pct": 0.47,
        "macd": 0.4, "signal": 0.2, "vol_z": 2.5,
        "sma_20": 315.0, "sma_50": 310.0, "sma_200": 300.0,
    }]}
    s = _summarize_quote("AAPL", raw)
    assert s["symbol"] == "AAPL"
    assert s["above_vwap"] is True
    assert s["pct_vs_vwap"] == round(100 * (320 - 316) / 316, 2)
    assert s["rsi_zone"] == "overbought"
    assert s["macd_state"] == "bullish"
    assert s["unusual_volume"] is True       # vol_z >= 2
    assert s["trend"] == "up"                # price > sma50 > sma200


def test_summarize_quote_handles_flat_fallback_quote():
    from src.agents.day_trader import _summarize_quote
    s = _summarize_quote("KO", {"price": 60.0, "source": "local-yfinance"})
    assert s["symbol"] == "KO" and s["price"] == 60.0


class ScriptedProvider:
    name = "scripted"
    model = "scripted"

    def __init__(self, results):
        self._queue = list(results)

    def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=1024):
        return self._queue.pop(0)


def _propose(symbol="AAPL", entry=150.0, stop=148.0):
    return ChatResult(text=None, tool_calls=[ToolCall(
        id="c1", name="propose_trade",
        arguments={"symbol": symbol, "entry_price": entry, "stop_price": stop,
                    "thesis": "test"})])


def test_kill_switch_halts_immediately(tmp_db, stub_bars):
    dbm.set_setting(tmp_db, "kill_switch", "on")
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN)
    out = agent.run_once()
    assert out.status == "halted" and out.error == "kill_switch"


def test_force_flat_closes_all_positions(tmp_db, stub_bars):
    stub_bars.set("AAPL", o=150, h=151, l=149, c=150)
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    # Seed a position by placing a buy first.
    from src.brokers.base import OrderRequest
    broker.place_order(OrderRequest(symbol="AAPL", side="buy", qty=10,
                                      sub_account="day", agent="manual"))
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=FORCE_FLAT)
    out = agent.run_once()
    assert out.status == "ok"
    # Position should now be flat.
    assert broker.list_positions("day") == []


def test_happy_path_proposes_sizes_and_places(tmp_db, stub_bars, monkeypatch):
    # Zero slippage/commission for predictable cash math.
    monkeypatch.setenv("SLIPPAGE_BPS", "0")
    monkeypatch.setenv("COMMISSION_BPS", "0")
    from src import config
    config.get_settings.cache_clear()

    stub_bars.set("AAPL", o=150, h=151, l=149, c=150)
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    fake = FakeShortTerm()
    # $30k equity * 1% risk = $300; stop $143 → $7/share → 42 shares by risk.
    # The 20% concentration cap ($6k / $150 = 40 shares) is tighter and binds.
    prov = ScriptedProvider([_propose("AAPL", 150.0, 143.0),
                              ChatResult(text="done")])
    agent = DayTraderAgent(tmp_db, broker, fake, provider=prov, now=MARKET_OPEN)
    out = agent.run_once()

    assert out.status == "ok"
    assert any(o["status"] == "filled" and o["symbol"] == "AAPL" for o in out.orders)
    pos = {p.symbol: p.qty for p in broker.list_positions("day")}
    # 20% concentration cap binds: $30k * 20% = $6,000 / $150 = 40 shares
    # (tighter than the 42 shares that 1%-risk sizing alone would allow).
    assert pos.get("AAPL") == 40.0


def test_concentration_cap_is_position_aware(tmp_db, stub_bars, monkeypatch):
    """Topping up an existing name cannot breach the per-symbol cap."""
    monkeypatch.setenv("SLIPPAGE_BPS", "0")
    monkeypatch.setenv("COMMISSION_BPS", "0")
    monkeypatch.setenv("STOCK_REC_MAX_ORDER_USD", "0")
    monkeypatch.setenv("STOCK_REC_MAX_SYMBOL_PCT", "0")
    monkeypatch.setenv("DAY_MAX_POSITION_PCT", "0.20")
    from src import config
    config.get_settings.cache_clear()

    stub_bars.set("AAPL", o=150, h=151, l=149, c=150)
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    from src.brokers.base import OrderRequest
    # Seed a position already at ~13% of $30k equity (26 sh * $150 = $3,900).
    broker.place_order(OrderRequest(symbol="AAPL", side="buy", qty=26,
                                     sub_account="day", agent="manual"))
    from src.agents.day_trader import _Proposal, DayTraderAgent
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN)
    equity = broker.get_account("day").equity
    ds = agent._validate_and_size(
        [_Proposal(symbol="AAPL", entry_price=150.0, stop_price=143.0,
                    side="buy", thesis="top up")])
    d = ds[0]
    # Cap is 20% ($6k). Already hold ~$3.9k, so only ~$2.1k (14 sh) more allowed.
    assert d.accepted
    total_notional = (26 + d.qty) * 150.0
    assert total_notional <= equity * 0.20 + 150.0
    config.get_settings.cache_clear()


def test_inverse_substitution_converts_short_leveraged_etf_to_long_inverse(tmp_db, stub_bars):
    """A short of a leveraged ETF is rewritten as a long of its inverse."""
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    fake = FakeShortTerm()  # lookup_ticker returns a flat quote priced at 150.0
    agent = DayTraderAgent(tmp_db, broker, fake, provider=ScriptedProvider([]),
                            now=MARKET_OPEN)
    from src.agents.day_trader import _Proposal
    # Short SOXL, entry 100, stop 105 → 5% stop distance.
    subbed = agent._maybe_substitute_inverse(
        [_Proposal(symbol="SOXL", entry_price=100.0, stop_price=105.0,
                    side="sell", thesis="bearish semis")])
    assert len(subbed) == 1
    p = subbed[0]
    assert p.symbol == "SOXS"          # inverse counterpart
    assert p.side == "buy"             # long the inverse
    assert p.entry_price == 150.0      # priced from the short-term client
    assert p.stop_price == 142.5       # 150 * (1 - 0.05), long stop below entry
    assert "inverse of short SOXL" in p.thesis


def test_inverse_substitution_leaves_ordinary_short_untouched(tmp_db, stub_bars):
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN)
    from src.agents.day_trader import _Proposal
    orig = [_Proposal(symbol="AAPL", entry_price=150.0, stop_price=153.0,
                       side="sell", thesis="short AAPL")]
    subbed = agent._maybe_substitute_inverse(orig)
    assert subbed[0].symbol == "AAPL" and subbed[0].side == "sell"


def test_summarize_ideas_ranks_and_flags_news_catalyst():
    from src.agents.day_trader import _summarize_ideas
    res = {"rows": [
        {"ticker": "SOXS", "direction": "long", "tier": "A", "heat_score": 88.5,
         "signal_tags": "atr_leader,gapper,premarket_mover", "rr": 2.0,
         "dollar_risk": 250.0, "dollar_gain": 500.0, "entry": 4.6, "stop": 4.5, "target": 4.8},
        {"ticker": "NVDA", "direction": "long", "tier": "A", "heat_score": 72.0,
         "signal_tags": "macd_bull_cross,news_spike", "rr": 2.0,
         "dollar_risk": 250.0, "dollar_gain": 500.0, "entry": 100.0, "stop": 98.0, "target": 104.0},
    ]}
    s = _summarize_ideas(res)
    assert s["count"] == 2
    assert s["ideas"][0]["ticker"] == "SOXS" and s["ideas"][1]["ticker"] == "NVDA"
    assert s["ideas"][0]["has_news_catalyst"] is False   # volatility-only
    assert s["ideas"][1]["has_news_catalyst"] is True    # news_spike fired
    assert "heat_score" in s["ideas"][0] and "dollar_risk" in s["ideas"][0]


def test_summarize_news_aggregates_sentiment():
    from src.agents.day_trader import _summarize_news
    raw = {"ticker": "AAPL", "articles": [
        {"headline": "AAPL soars on blowout earnings", "sentiment": {"score": 0.8, "label": "Positive"}, "source": "finnhub"},
        {"headline": "Analysts raise targets", "sentiment": {"score": 0.4, "label": "Positive"}, "source": "yahoo"},
        {"headline": "Minor supply concern", "sentiment": {"score": -0.5, "label": "Negative"}, "source": "reddit"},
    ]}
    s = _summarize_news("AAPL", raw)
    assert s["count"] == 3
    assert s["sentiment_label"] == "positive"      # avg (0.8+0.4-0.5)/3 = 0.233
    assert s["bullish_articles"] == 2 and s["bearish_articles"] == 1
    assert len(s["top_headlines"]) == 3
    assert s["top_headlines"][0]["headline"].startswith("AAPL soars")  # highest |score|


def test_summarize_news_handles_no_articles():
    from src.agents.day_trader import _summarize_news
    s = _summarize_news("KO", {"ticker": "KO", "articles": []})
    assert s["count"] == 0 and s["sentiment_label"] == "no_news"


def test_summarize_analyst_extracts_rating_and_upside():
    from src.agents.day_trader import _summarize_analyst
    raw = {"ticker": "NVDA", "price": 100.0, "target_price": 130.0,
           "rating": "Strong Buy", "analyst_count": 42,
           "sentiment_score": 0.3, "sentiment_label": "Positive"}
    s = _summarize_analyst("NVDA", raw)
    assert s["rating"] == "Strong Buy"
    assert s["target_price"] == 130.0
    assert s["upside_pct"] == 30.0                 # (130-100)/100 * 100
    assert s["analyst_count"] == 42


class FakeLongTerm:
    """Stand-in exposing the analyst/news methods the day agent calls."""

    def lookup_ticker(self, ticker):
        return {"ticker": ticker, "price": 100.0, "target_price": 120.0,
                "rating": "Buy", "analyst_count": 20, "upside_pct": 20.0,
                "sentiment_score": 0.2, "sentiment_label": "Positive"}

    def get_news(self, ticker, *, days=7, limit=20):
        return {"ticker": ticker, "articles": [
            {"headline": f"{ticker} steady", "sentiment": {"score": 0.1, "label": "Positive"},
             "source": "finnhub"}]}


def test_day_agent_get_analyst_view_via_long_term(tmp_db, stub_bars):
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN,
                            long_term=FakeLongTerm())
    # Build the tool closures and invoke get_analyst_view / get_news directly.
    view = agent._analyst_view("NVDA")
    assert view["rating"] == "Buy" and view["upside_pct"] == 20.0
    news = agent._news("NVDA", days=2)
    assert news["sentiment_label"] == "positive" and news["count"] == 1


def test_concentration_cap_limits_single_name_notional(tmp_db, stub_bars, monkeypatch):
    monkeypatch.setenv("SLIPPAGE_BPS", "0")
    monkeypatch.setenv("COMMISSION_BPS", "0")
    monkeypatch.setenv("STOCK_REC_MAX_ORDER_USD", "0")     # venue caps disabled
    monkeypatch.setenv("STOCK_REC_MAX_SYMBOL_PCT", "0")
    monkeypatch.setenv("DAY_MAX_POSITION_PCT", "0.20")
    from src import config
    config.get_settings.cache_clear()

    # Cheap volatile name with a tight stop: 1%-risk sizing alone would buy a
    # huge notional; the 20% concentration cap must bound it.
    stub_bars.set("SOXS", o=4.0, h=4.1, l=3.9, c=4.0)
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    from src.agents.day_trader import _Proposal
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN)
    equity = broker.get_account("day").equity
    ds = agent._validate_and_size(
        [_Proposal(symbol="SOXS", entry_price=4.0, stop_price=3.99, side="buy",
                    thesis="cheap tight stop")])
    d = ds[0]
    assert d.accepted
    # Notional must not exceed 20% of equity (+ one share of rounding slack).
    assert d.qty * 4.0 <= equity * 0.20 + 4.0
    config.get_settings.cache_clear()


class FakeOptions:
    """Stand-in for AlpacaOptions."""

    def __init__(self, fail=False):
        self.fail = fail
        self.orders = []

    def options_enabled(self):
        return True

    def find_contracts(self, underlying, *, option_type="call", limit=15):
        return [{"symbol": f"{underlying}250620C00190000", "underlying": underlying,
                 "type": option_type, "strike": 190.0, "expiration": "2025-06-20",
                 "close_price": 3.20}]

    def place_order(self, *, occ_symbol, qty, side, order_type="market",
                    limit_price=None, time_in_force="day"):
        self.orders.append({"occ": occ_symbol, "qty": qty, "side": side})
        if self.fail:
            raise RuntimeError("alpaca options outage")
        return {"id": "opt-xyz", "status": "accepted", "symbol": occ_symbol}


def _propose_option(occ="AAPL250620C00190000", qty=1, side="buy"):
    return ChatResult(text=None, tool_calls=[ToolCall(
        id="o1", name="propose_option",
        arguments={"occ_symbol": occ, "qty": qty, "side": side, "thesis": "bullish"})])


def test_option_order_placed_and_recorded(tmp_db, stub_bars):
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    opts = FakeOptions()
    prov = ScriptedProvider([_propose_option(), ChatResult(text="done")])
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(), provider=prov,
                            now=MARKET_OPEN, options=opts)
    out = agent.run_once()
    assert out.status == "ok"
    assert len(opts.orders) == 1 and opts.orders[0]["occ"] == "AAPL250620C00190000"
    row = tmp_db.execute(
        "SELECT symbol, status, external_id, venue FROM orders WHERE venue='alpaca_options'"
    ).fetchone()
    assert row["symbol"] == "AAPL250620C00190000"
    assert row["status"] == "accepted" and row["external_id"] == "opt-xyz"


def test_option_order_failure_recorded_as_rejected(tmp_db, stub_bars):
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    opts = FakeOptions(fail=True)
    prov = ScriptedProvider([_propose_option(), ChatResult(text="done")])
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(), provider=prov,
                            now=MARKET_OPEN, options=opts)
    out = agent.run_once()
    assert out.status == "ok"
    row = tmp_db.execute(
        "SELECT status FROM orders WHERE venue='alpaca_options'"
    ).fetchone()
    assert row["status"] == "rejected"


def test_no_options_client_runs_fine(tmp_db, stub_bars):
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    prov = ScriptedProvider([ChatResult(text="no trades today")])
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(), provider=prov, now=MARKET_OPEN)
    out = agent.run_once()
    assert out.status == "ok"


def test_max_concurrent_positions_rejects_extra(tmp_db, stub_bars, monkeypatch):
    monkeypatch.setenv("SLIPPAGE_BPS", "0")
    monkeypatch.setenv("COMMISSION_BPS", "0")
    from src import config
    config.get_settings.cache_clear()

    # Seed 5 distinct positions on the day book.
    for s in ("AAA", "BBB", "CCC", "DDD", "EEE"):
        stub_bars.set(s, o=10, h=10.5, l=9.5, c=10)
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    from src.brokers.base import OrderRequest
    for s in ("AAA", "BBB", "CCC", "DDD", "EEE"):
        broker.place_order(OrderRequest(symbol=s, side="buy", qty=1, sub_account="day",
                                          agent="manual"))
    # Now the LLM proposes a 6th.
    stub_bars.set("FFF", o=10, h=10.5, l=9.5, c=10)
    prov = ScriptedProvider([_propose("FFF", 10.0, 9.0), ChatResult(text="done")])
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(), provider=prov, now=MARKET_OPEN)
    out = agent.run_once()

    # No 6th fill.
    assert not any(o["symbol"] == "FFF" and o["status"] == "filled" for o in out.orders)
    # Decision recorded with reject_reason.
    rejected = [d for d in out.decisions if d["symbol"] == "FFF" and not d["accepted"]]
    assert rejected and rejected[0]["reject_reason"] == "max_concurrent_positions"


def test_daily_drawdown_halts(tmp_db, stub_bars):
    # Seed an equity_curve row that pins starting equity high, then current equity is low.
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    aid = dbm.get_account_id(tmp_db, "day")
    today = MARKET_OPEN.replace(hour=0, minute=0).isoformat()
    tmp_db.execute(
        "INSERT INTO equity_curve(account_id, ts, cash, positions_value, equity) "
        "VALUES (?, ?, ?, ?, ?)",
        (aid, today, 30_000.0, 0.0, 30_000.0),
    )
    # Bleed cash so current equity is < -2%.
    tmp_db.execute(
        "INSERT INTO cash_ledger(account_id, ts, delta, reason) VALUES (?, ?, ?, 'fee')",
        (aid, today, -1_000.0),
    )
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN)
    out = agent.run_once()
    assert out.status == "halted" and out.error == "daily_drawdown"
