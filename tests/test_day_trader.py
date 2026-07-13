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
    # Wider stop so 1% risk sizing stays under the 25% per-symbol cap.
    # $30k equity * 1% risk = $300; stop $143 → $7/share → 42 shares = $6.3k (21%).
    prov = ScriptedProvider([_propose("AAPL", 150.0, 143.0),
                              ChatResult(text="done")])
    agent = DayTraderAgent(tmp_db, broker, fake, provider=prov, now=MARKET_OPEN)
    out = agent.run_once()

    assert out.status == "ok"
    assert any(o["status"] == "filled" and o["symbol"] == "AAPL" for o in out.orders)
    pos = {p.symbol: p.qty for p in broker.list_positions("day")}
    assert pos.get("AAPL") == 42.0


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
