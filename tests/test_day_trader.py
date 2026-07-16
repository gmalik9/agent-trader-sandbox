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


def _exit(symbol="AAPL", reason="rotate"):
    return ChatResult(text=None, tool_calls=[ToolCall(
        id="e1", name="exit_position",
        arguments={"symbol": symbol, "reason": reason})])


def _ideas_call():
    return ChatResult(text=None, tool_calls=[ToolCall(
        id="i1", name="list_intraday_ideas", arguments={"tier": "A", "limit": 5})])


def test_agent_can_proactively_exit_position(tmp_db, stub_bars, monkeypatch):
    """The agent can close a held position via the exit_position tool."""
    monkeypatch.setenv("SLIPPAGE_BPS", "0")
    monkeypatch.setenv("COMMISSION_BPS", "0")
    from src import config
    config.get_settings.cache_clear()
    stub_bars.set("AAPL", o=150, h=151, l=149, c=150)
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    from src.brokers.base import OrderRequest
    broker.place_order(OrderRequest(symbol="AAPL", side="buy", qty=10,
                                     sub_account="day", agent="manual"))
    assert broker.list_positions("day")                    # held before
    prov = ScriptedProvider([_exit("AAPL", "rotate"), ChatResult(text="done")])
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(), provider=prov, now=MARKET_OPEN)
    out = agent.run_once()
    assert out.status == "ok"
    assert broker.list_positions("day") == []              # agent closed it
    assert any("agent_exit" in d for d in out.decisions)   # recorded
    config.get_settings.cache_clear()


def test_is_option_symbol_detects_occ():
    from src.agents.day_trader import _is_option_symbol
    assert _is_option_symbol("F260717C00016000")      # OCC option
    assert _is_option_symbol("AAPL250620C00190000")
    assert not _is_option_symbol("AAPL")              # stock
    assert not _is_option_symbol("SOXS")              # ETF


def test_exit_position_refuses_option_symbol(tmp_db, stub_bars):
    """The agent must not try to close an option via the stock endpoint."""
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    prov = ScriptedProvider([_exit("F260717C00016000", "worthless"), ChatResult(text="done")])
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(), provider=prov, now=MARKET_OPEN)
    out = agent.run_once()
    assert out.status == "ok"
    # No agent_exit recorded for the option (it was refused, not closed).
    assert not any("agent_exit" in d for d in out.decisions)


def test_compact_mode_toggle_via_setting(tmp_db, stub_bars):
    """The compact_prompt setting flips _compact_mode; default (unset) is False."""
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN)
    assert agent._compact_mode() is False           # unset → config default (False)
    dbm.set_setting(tmp_db, "compact_prompt", "on")
    assert agent._compact_mode() is True            # toggle ON
    dbm.set_setting(tmp_db, "compact_prompt", "off")
    assert agent._compact_mode() is False           # toggle OFF


def test_compact_ideas_result_trims_payload():
    from src.agents.day_trader import _compact_ideas_result
    full = {
        "count": 1,
        "ideas": [
            {"ticker": "AAA", "direction": "long", "tier": "A", "heat_score": 90.0,
             "has_news_catalyst": True, "entry": 10.0, "stop": 9.5, "target": 11.0,
             "rr": 2.0, "signal_tags": "x,y,z", "dollar_risk": 250, "already_held_pct": 0.0,
             "room_left": 20.0, "at_cap": False, "theme": "t", "effective_symbol": "AAA"},
        ],
        "portfolio": {"equity": 100000, "cash": 5000, "pct_cash_idle": 5.0,
                       "open_position_count": 8, "max_positions": 8, "book_full": True,
                       "held": {"AAA": 20.0}, "theme_exposure_pct": {"t": 20.0},
                       "holdings": [{"symbol": "AAA", "side": "long", "pct_of_equity": 20.0,
                                      "unrealized_pnl": -100.0, "avg_cost": 10.0, "mark": 9.8,
                                      "stop": 9.5, "pct_to_stop": 3.0, "theme": "t"}]},
    }
    slim = _compact_ideas_result(full)
    i = slim["ideas"][0]
    assert set(i) == {"ticker", "direction", "tier", "heat_score", "has_news_catalyst",
                       "entry", "stop", "target", "rr"}
    assert slim["portfolio"]["book_full"] is True
    h = slim["portfolio"]["holdings"][0]
    assert set(h) == {"symbol", "side", "unrealized_pnl", "pct_to_stop"}


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
    monkeypatch.setenv("DAY_REQUIRE_DILIGENCE", "false")  # sizing test, not diligence
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


def test_theme_cap_rejects_over_concentrated_correlated_names(tmp_db, stub_bars, monkeypatch):
    """Once a correlation theme is full, another name in it is rejected."""
    monkeypatch.setenv("SLIPPAGE_BPS", "0")
    monkeypatch.setenv("COMMISSION_BPS", "0")
    monkeypatch.setenv("STOCK_REC_MAX_ORDER_USD", "0")
    monkeypatch.setenv("STOCK_REC_MAX_SYMBOL_PCT", "0")
    monkeypatch.setenv("DAY_MAX_POSITION_PCT", "0.20")
    monkeypatch.setenv("DAY_THEME_MAX_PCT", "0.35")
    from src import config
    config.get_settings.cache_clear()

    # $30k equity, semis theme cap = 35% = $10.5k. Seed SOXL AT the cap.
    stub_bars.set("SOXL", o=100, h=101, l=99, c=100)
    stub_bars.set("NVDL", o=50, h=51, l=49, c=50)
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars, mirror=True)  # no sandbox caps
    from src.brokers.base import OrderRequest
    broker.place_order(OrderRequest(symbol="SOXL", side="buy", qty=105,
                                     sub_account="day", agent="manual"))  # $10.5k = 35%
    from src.agents.day_trader import _Proposal, DayTraderAgent
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN)
    # NVDL is the same 'semis' theme → the theme is essentially full → reject.
    ds = agent._validate_and_size(
        [_Proposal(symbol="NVDL", entry_price=50.0, stop_price=49.0,
                    side="buy", thesis="more semis")])
    assert not ds[0].accepted
    assert ds[0].reject_reason.startswith("theme_at_cap")
    config.get_settings.cache_clear()


def test_dust_positions_do_not_consume_slots(tmp_db, stub_bars, monkeypatch):
    """Trivial leftover positions (<2% of equity) must not block new names."""
    monkeypatch.setenv("SLIPPAGE_BPS", "0")
    monkeypatch.setenv("COMMISSION_BPS", "0")
    from src import config
    config.get_settings.cache_clear()
    # 8 dust positions (1 share * $10 = 0.03% of $30k equity each).
    dust = ("AAA", "BBB", "CCC", "DDD", "EEE", "GGG", "HHH", "III")
    for s in dust:
        stub_bars.set(s, o=10, h=10.5, l=9.5, c=10)
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    from src.brokers.base import OrderRequest
    for s in dust:
        broker.place_order(OrderRequest(symbol=s, side="buy", qty=1,
                                         sub_account="day", agent="manual"))
    stub_bars.set("NEW", o=100, h=101, l=99, c=100)
    from src.agents.day_trader import _Proposal, DayTraderAgent
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN)
    ds = agent._validate_and_size(
        [_Proposal(symbol="NEW", entry_price=100.0, stop_price=98.0,
                    side="buy", thesis="fresh name")])
    # Dust doesn't count → the new name is accepted despite 8 held symbols.
    assert ds[0].accepted and ds[0].reject_reason != "max_concurrent_positions"
    config.get_settings.cache_clear()


def test_summarize_ideas_annotates_holdings_and_room():
    from src.agents.day_trader import _summarize_ideas
    res = {"rows": [
        {"ticker": "SOXS", "direction": "long", "tier": "A", "heat_score": 88.0,
         "signal_tags": "atr_leader", "rr": 2.0, "entry": 4.0, "stop": 3.9, "target": 4.2},
        {"ticker": "NVDA", "direction": "long", "tier": "A", "heat_score": 70.0,
         "signal_tags": "macd_bull_cross", "rr": 2.0, "entry": 100.0, "stop": 98.0, "target": 104.0},
    ]}
    s = _summarize_ideas(res, held_pct={"SOXS": 20.0}, cap_pct=20.0,
                          context={"pct_cash_idle": 79.0})
    by = {i["ticker"]: i for i in s["ideas"]}
    # At-cap names are hidden from the tradable list (forces rotation)…
    assert "SOXS" not in by
    assert s["hidden_at_name_cap"] == ["SOXS"]
    # …and names with room remain, annotated.
    assert by["NVDA"]["at_cap"] is False and by["NVDA"]["room_left"] == 20.0
    assert s["count"] == 1
    assert s["portfolio"]["pct_cash_idle"] == 79.0


def test_summarize_ideas_hides_inverse_of_capped_holding():
    """A short idea that substitutes into an at-cap holding is hidden too."""
    from src.agents.day_trader import _summarize_ideas
    # We hold UCO at its 20% cap. "short SCO" would convert to "long UCO" — so
    # the SCO idea must be hidden even though we don't hold SCO directly.
    res = {"rows": [
        {"ticker": "SCO", "direction": "short", "tier": "A", "heat_score": 80.0,
         "signal_tags": "gapper", "rr": 2.0, "entry": 20.0, "stop": 21.0, "target": 18.0},
        {"ticker": "TECS", "direction": "long", "tier": "A", "heat_score": 70.0,
         "signal_tags": "vwap_reclaim", "rr": 2.0, "entry": 7.0, "stop": 6.9, "target": 7.2},
    ]}
    s = _summarize_ideas(res, held_pct={"UCO": 20.0}, cap_pct=20.0)
    tickers = [i["ticker"] for i in s["ideas"]]
    assert "SCO" not in tickers          # short SCO -> long UCO (capped) → hidden
    assert "TECS" in tickers
    assert "SCO" in s["hidden_at_name_cap"]


def test_summarize_ideas_hides_theme_at_cap_and_cooldown():
    """Names in a full correlation theme or on cooldown are hidden."""
    from src.agents.day_trader import _summarize_ideas
    res = {"rows": [
        # semis theme is full (SOXL held via theme_exposure) → NVDL hidden
        {"ticker": "NVDL", "direction": "long", "tier": "A", "heat_score": 80.0,
         "signal_tags": "gapper", "rr": 2.0, "entry": 40.0, "stop": 39.0, "target": 42.0},
        # on cooldown → hidden
        {"ticker": "AAPL", "direction": "long", "tier": "A", "heat_score": 70.0,
         "signal_tags": "vwap_reclaim", "rr": 2.0, "entry": 200.0, "stop": 196.0, "target": 208.0},
        # clean, different theme → tradable
        {"ticker": "XLE", "direction": "long", "tier": "A", "heat_score": 60.0,
         "signal_tags": "macd_bull_cross", "rr": 2.0, "entry": 90.0, "stop": 88.0, "target": 94.0},
    ]}
    s = _summarize_ideas(res, held_pct={}, cap_pct=20.0,
                          theme_exposure={"semis": 36.0}, theme_cap_pct=35.0,
                          cooldown={"AAPL"})
    tickers = [i["ticker"] for i in s["ideas"]]
    assert tickers == ["XLE"]
    assert "NVDL" in s["hidden_theme_at_cap"]
    assert "AAPL" in s["hidden_on_cooldown"]


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


def test_reprice_to_market_rescales_stale_scanner_price(tmp_db, stub_bars):
    """A stale scanner entry (10x off) is rescaled to the real market price,
    preserving the % stop distance."""
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    # FakeShortTerm.lookup_ticker reports price/last = 150.0 for any symbol.
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN)
    from src.agents.day_trader import _Proposal
    # Scanner says entry 15, stop 14.7 (2% stop). Real price is 150 (10x).
    out = agent._reprice_to_market(
        [_Proposal(symbol="XYZ", entry_price=15.0, stop_price=14.7, side="buy")])
    p = out[0]
    assert p.entry_price == 150.0                     # repriced to market
    assert abs(p.stop_price - 147.0) < 0.01           # stop scaled x10 (2% preserved)
    assert "repriced" in p.thesis


def test_reprice_to_market_leaves_accurate_price_untouched(tmp_db, stub_bars):
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN)
    from src.agents.day_trader import _Proposal
    # Scanner entry 150 already matches the real price (150) → unchanged.
    out = agent._reprice_to_market(
        [_Proposal(symbol="XYZ", entry_price=150.0, stop_price=147.0, side="buy")])
    assert out[0].entry_price == 150.0 and "repriced" not in (out[0].thesis or "")


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
    monkeypatch.setenv("DAY_DEFAULT_STOP_PCT", "0")   # don't auto-close the seeded book
    monkeypatch.setenv("DAY_REQUIRE_DILIGENCE", "false")
    from src import config
    config.get_settings.cache_clear()

    # Seed 8 distinct MATERIAL positions (each ≥2% of equity so they consume a
    # concurrent-position slot). $30k equity → 2% = $600; use $1,000 each.
    syms = ("AAA", "BBB", "CCC", "DDD", "EEE", "GGG", "HHH", "III")
    for s in syms:
        stub_bars.set(s, o=10, h=10.5, l=9.5, c=10)
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    from src.brokers.base import OrderRequest
    for s in syms:
        broker.place_order(OrderRequest(symbol=s, side="buy", qty=100, sub_account="day",
                                          agent="manual"))
    # Now the LLM proposes a 9th (over the 8-position limit).
    stub_bars.set("FFF", o=10, h=10.5, l=9.5, c=10)
    prov = ScriptedProvider([_propose("FFF", 10.0, 9.0), ChatResult(text="done")])
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(), provider=prov, now=MARKET_OPEN)
    out = agent.run_once()

    # No 9th fill.
    assert not any(o["symbol"] == "FFF" and o["status"] == "filled" for o in out.orders)
    # Decision recorded with reject_reason.
    rejected = [d for d in out.decisions if d.get("symbol") == "FFF" and not d.get("accepted")]
    assert rejected and rejected[0]["reject_reason"] == "max_concurrent_positions"


def test_idea_context_surfaces_holdings_and_book_full(tmp_db, stub_bars, monkeypatch):
    """list_intraday_ideas' portfolio block exposes per-holding detail and a
    book_full flag so the agent can rotate."""
    import json
    monkeypatch.setenv("SLIPPAGE_BPS", "0")
    monkeypatch.setenv("COMMISSION_BPS", "0")
    monkeypatch.setenv("DAY_DEFAULT_STOP_PCT", "0")
    from src import config
    config.get_settings.cache_clear()

    # Seed 8 material positions so the book is full.
    syms = ("AAA", "BBB", "CCC", "DDD", "EEE", "GGG", "HHH", "III")
    for s in syms:
        stub_bars.set(s, o=10, h=10.5, l=9.5, c=10)
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars, mirror=True)
    from src.brokers.base import OrderRequest
    for s in syms:
        broker.place_order(OrderRequest(symbol=s, side="buy", qty=100,
                                         sub_account="day", agent="manual"))
    # Give AAA an explicit stop plan so pct_to_stop is populated.
    aid = dbm.get_account_id(tmp_db, "day")
    dbm.upsert_position_plan(tmp_db, account_id=aid, symbol="AAA", side="buy",
                             entry_price=10.0, stop_price=9.5, target_price=11.0,
                             agent="day")
    prov = ScriptedProvider([_ideas_call(), ChatResult(text="done")])
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(), provider=prov, now=MARKET_OPEN)
    # Keep the stop monitor from closing AAA (FakeShortTerm would report 150).
    monkeypatch.setattr(agent, "_latest_price", lambda s: 10.0)
    out = agent.run_once()
    assert out.status == "ok"
    # Read the list_intraday_ideas tool result from the recorded run.
    row = tmp_db.execute(
        "SELECT tools_called FROM agent_runs WHERE agent='day' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    steps = json.loads(row["tools_called"] or "[]")
    result = None
    for st in steps:
        for tc in st.get("tool_calls", []):
            if tc.get("name") == "list_intraday_ideas":
                result = tc.get("result")
    assert result is not None
    pf = result["portfolio"]
    assert pf["book_full"] is True
    assert pf["open_position_count"] == 8
    holdings = {h["symbol"]: h for h in pf["holdings"]}
    assert "AAA" in holdings and holdings["AAA"]["pct_to_stop"] is not None
    assert holdings["AAA"]["side"] == "long"
    config.get_settings.cache_clear()


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


def test_stop_monitor_closes_long_when_stop_breached(tmp_db, stub_bars, monkeypatch):
    """A held long whose live price has fallen through its stop is closed."""
    stub_bars.set("AAA", o=100, h=101, l=99, c=100)
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    from src.brokers.base import OrderRequest
    broker.place_order(OrderRequest(symbol="AAA", side="buy", qty=50,
                                     sub_account="day", agent="manual"))
    aid = dbm.get_account_id(tmp_db, "day")
    dbm.upsert_position_plan(tmp_db, account_id=aid, symbol="AAA", side="buy",
                             entry_price=100.0, stop_price=95.0, target_price=110.0,
                             agent="day")
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN)
    # Live price 94 is below the 95 stop → should close.
    monkeypatch.setattr(agent, "_latest_price", lambda s: 94.0)
    actions = agent._manage_open_positions()
    assert actions and actions[0]["reason"] == "stop_hit"
    assert broker.list_positions("day") == []            # position closed
    assert not dbm.get_active_position_plans(tmp_db, aid)  # plan retired


def test_stop_monitor_holds_position_above_stop(tmp_db, stub_bars, monkeypatch):
    """A held long still above its stop (and below target) is left alone."""
    stub_bars.set("AAA", o=100, h=101, l=99, c=100)
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    from src.brokers.base import OrderRequest
    broker.place_order(OrderRequest(symbol="AAA", side="buy", qty=50,
                                     sub_account="day", agent="manual"))
    aid = dbm.get_account_id(tmp_db, "day")
    dbm.upsert_position_plan(tmp_db, account_id=aid, symbol="AAA", side="buy",
                             entry_price=100.0, stop_price=95.0, target_price=110.0,
                             agent="day")
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN)
    monkeypatch.setattr(agent, "_latest_price", lambda s: 101.0)  # between stop & target
    actions = agent._manage_open_positions()
    assert actions == []
    assert broker.list_positions("day")                  # still open
    assert dbm.get_active_position_plans(tmp_db, aid)     # plan still active


def test_stop_monitor_takes_profit_at_target(tmp_db, stub_bars, monkeypatch):
    """A held long that reaches its target is closed (take-profit)."""
    stub_bars.set("AAA", o=100, h=101, l=99, c=100)
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    from src.brokers.base import OrderRequest
    broker.place_order(OrderRequest(symbol="AAA", side="buy", qty=50,
                                     sub_account="day", agent="manual"))
    aid = dbm.get_account_id(tmp_db, "day")
    dbm.upsert_position_plan(tmp_db, account_id=aid, symbol="AAA", side="buy",
                             entry_price=100.0, stop_price=95.0, target_price=110.0,
                             agent="day")
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN)
    monkeypatch.setattr(agent, "_latest_price", lambda s: 111.0)  # above target
    actions = agent._manage_open_positions()
    assert actions and actions[0]["reason"] == "target_hit"
    assert broker.list_positions("day") == []


def test_default_stop_backfills_plan_for_unplanned_position(tmp_db, stub_bars, monkeypatch):
    """With DAY_DEFAULT_STOP_PCT>0, a held position lacking a plan gets one."""
    monkeypatch.setenv("SLIPPAGE_BPS", "0")
    monkeypatch.setenv("COMMISSION_BPS", "0")
    monkeypatch.setenv("DAY_DEFAULT_STOP_PCT", "0.05")
    from src import config
    config.get_settings.cache_clear()
    stub_bars.set("AAA", o=100, h=101, l=99, c=100)
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    from src.brokers.base import OrderRequest
    broker.place_order(OrderRequest(symbol="AAA", side="buy", qty=50,
                                     sub_account="day", agent="manual"))
    aid = dbm.get_account_id(tmp_db, "day")
    assert not dbm.get_active_position_plans(tmp_db, aid)  # no plan yet
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN)
    monkeypatch.setattr(agent, "_latest_price", lambda s: 100.0)  # at entry, no breach
    agent._manage_open_positions()
    plans = dbm.get_active_position_plans(tmp_db, aid)
    assert len(plans) == 1
    assert plans[0]["symbol"] == "AAA" and plans[0]["side"] == "buy"
    assert abs(plans[0]["stop_price"] - 95.0) < 0.01   # 5% below the $100 avg cost
    config.get_settings.cache_clear()


def test_default_stop_off_by_default(tmp_db, stub_bars, monkeypatch):
    """With the default (0), no plan is fabricated for an unplanned position."""
    monkeypatch.setenv("DAY_DEFAULT_STOP_PCT", "0")
    from src import config
    config.get_settings.cache_clear()
    stub_bars.set("AAA", o=100, h=101, l=99, c=100)
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    from src.brokers.base import OrderRequest
    broker.place_order(OrderRequest(symbol="AAA", side="buy", qty=50,
                                     sub_account="day", agent="manual"))
    aid = dbm.get_account_id(tmp_db, "day")
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN)
    agent._manage_open_positions()
    assert not dbm.get_active_position_plans(tmp_db, aid)  # nothing fabricated
    config.get_settings.cache_clear()


def test_stop_monitor_retires_plan_when_flat(tmp_db, stub_bars):
    """A plan for a symbol no longer held is retired without any close."""
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    aid = dbm.get_account_id(tmp_db, "day")
    dbm.upsert_position_plan(tmp_db, account_id=aid, symbol="GONE", side="buy",
                             entry_price=100.0, stop_price=95.0, target_price=110.0,
                             agent="day")
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN)
    actions = agent._manage_open_positions()
    assert actions == []
    assert not dbm.get_active_position_plans(tmp_db, aid)  # stale plan cleared


def test_diligence_gate_rejects_undiligenced_proposal(tmp_db, stub_bars, monkeypatch):
    """A proposal whose symbol skipped get_news/get_analyst_view is rejected."""
    monkeypatch.setenv("DAY_REQUIRE_DILIGENCE", "true")
    from src import config
    config.get_settings.cache_clear()
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars, mirror=True)
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN)
    from src.agents.day_trader import _Proposal
    # Tracking active this tick, but AAPL was never diligenced.
    agent._news_checked = set()
    agent._analyst_checked = set()
    ds = agent._validate_and_size(
        [_Proposal(symbol="AAPL", entry_price=150.0, stop_price=143.0,
                    side="buy", thesis="no homework")])
    assert not ds[0].accepted
    assert ds[0].reject_reason == "insufficient_diligence"
    config.get_settings.cache_clear()


def test_diligence_gate_allows_fully_diligenced_proposal(tmp_db, stub_bars, monkeypatch):
    """A proposal whose symbol got both news + analyst checks passes the gate."""
    monkeypatch.setenv("SLIPPAGE_BPS", "0")
    monkeypatch.setenv("COMMISSION_BPS", "0")
    monkeypatch.setenv("DAY_REQUIRE_DILIGENCE", "true")
    from src import config
    config.get_settings.cache_clear()
    stub_bars.set("AAPL", o=150, h=151, l=149, c=150)
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars, mirror=True)
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(),
                            provider=ScriptedProvider([]), now=MARKET_OPEN)
    from src.agents.day_trader import _Proposal
    agent._news_checked = {"AAPL"}
    agent._analyst_checked = {"AAPL"}
    ds = agent._validate_and_size(
        [_Proposal(symbol="AAPL", entry_price=150.0, stop_price=143.0,
                    side="buy", thesis="did homework")])
    assert ds[0].accepted and ds[0].reject_reason != "insufficient_diligence"
    config.get_settings.cache_clear()


def test_place_records_stop_plan(tmp_db, stub_bars, monkeypatch):
    """Placing a sized trade persists an active stop plan the monitor can use."""
    monkeypatch.setenv("SLIPPAGE_BPS", "0")
    monkeypatch.setenv("COMMISSION_BPS", "0")
    monkeypatch.setenv("DAY_REQUIRE_DILIGENCE", "false")
    from src import config
    config.get_settings.cache_clear()
    stub_bars.set("AAPL", o=150, h=151, l=149, c=150)
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    prov = ScriptedProvider([_propose("AAPL", 150.0, 143.0), ChatResult(text="done")])
    agent = DayTraderAgent(tmp_db, broker, FakeShortTerm(), provider=prov, now=MARKET_OPEN)
    out = agent.run_once()
    assert out.status == "ok"
    aid = dbm.get_account_id(tmp_db, "day")
    plans = dbm.get_active_position_plans(tmp_db, aid)
    assert len(plans) == 1
    assert plans[0]["symbol"] == "AAPL" and plans[0]["stop_price"] == 143.0
    config.get_settings.cache_clear()

