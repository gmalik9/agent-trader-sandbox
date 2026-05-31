from __future__ import annotations

from datetime import datetime, timezone

from src.agents.long_term import LongTermAgent
from src.brokers.sandbox_broker import SandboxBroker
from src.llm.provider import ChatResult, ToolCall


CLOCK = datetime(2025, 1, 14, 21, 30, tzinfo=timezone.utc)  # 16:30 ET


class FakeLongTerm:
    def __init__(self) -> None:
        self.calls = []

    def get_recommendations(self, **kw):
        return {"recommendations": [{"symbol": "MSFT", "weight": 0.25}]}

    def lookup_ticker(self, symbol):
        return {"symbol": symbol, "price": 400.0, "last": 400.0}

    def get_news(self, symbol, **kw):
        return {"news": []}


class ScriptedProvider:
    name = "scripted"
    model = "scripted"

    def __init__(self, results):
        self._queue = list(results)

    def chat(self, messages, **kw):
        return self._queue.pop(0)


def _propose(symbol="MSFT", side="buy", weight=10.0):
    return ChatResult(text=None, tool_calls=[ToolCall(
        id="c1", name="propose_rebalance",
        arguments={"symbol": symbol, "side": side,
                    "target_weight_pct": weight, "thesis": "test"})])


def test_kill_switch_halts(tmp_db, stub_bars):
    from src.sandbox import db as dbm
    dbm.set_setting(tmp_db, "kill_switch", "on")
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    agent = LongTermAgent(tmp_db, broker, FakeLongTerm(),
                           provider=ScriptedProvider([]), now=CLOCK)
    out = agent.run_once()
    assert out.status == "halted"


def test_rebalance_buy_sizes_to_target_weight(tmp_db, stub_bars, monkeypatch):
    monkeypatch.setenv("SLIPPAGE_BPS", "0")
    monkeypatch.setenv("COMMISSION_BPS", "0")
    from src import config
    config.get_settings.cache_clear()

    stub_bars.set("MSFT", o=400, h=401, l=399, c=400)
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    prov = ScriptedProvider([_propose("MSFT", "buy", 10.0), ChatResult(text="done")])
    agent = LongTermAgent(tmp_db, broker, FakeLongTerm(), provider=prov, now=CLOCK)
    out = agent.run_once()

    # Long sub-account equity = $70k; 10% target = $7k; $7k / $400 = 17.
    assert any(o["symbol"] == "MSFT" and o["status"] == "filled" for o in out.orders)
    pos = {p.symbol: p.qty for p in broker.list_positions("long")}
    assert pos.get("MSFT") == 17.0


def test_target_clamped_to_max_25_pct(tmp_db, stub_bars, monkeypatch):
    monkeypatch.setenv("SLIPPAGE_BPS", "0")
    monkeypatch.setenv("COMMISSION_BPS", "0")
    from src import config
    config.get_settings.cache_clear()

    stub_bars.set("MSFT", o=400, h=401, l=399, c=400)
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    # LLM asks for 80% — should be clamped to 25%.
    prov = ScriptedProvider([_propose("MSFT", "buy", 80.0), ChatResult(text="done")])
    agent = LongTermAgent(tmp_db, broker, FakeLongTerm(), provider=prov, now=CLOCK)
    out = agent.run_once()
    pos = {p.symbol: p.qty for p in broker.list_positions("long")}
    # 25% of $70k = $17.5k; $17.5k / $400 = 43.
    assert pos.get("MSFT") == 43.0


def test_throttle_skips_when_recent_run_no_drift(tmp_db, stub_bars):
    from src.sandbox import db as dbm
    aid = dbm.get_account_id(tmp_db, "long")
    # Pretend we already ran successfully today.
    tmp_db.execute(
        "INSERT INTO agent_runs(account_id, ts, agent, status, prompt) "
        "VALUES (?, ?, 'long', 'ok', '')",
        (aid, CLOCK.isoformat()),
    )
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    # No positions → drift considered "always significant" so we WOULDN'T throttle…
    # Add a small position so drift check sees "no significant drift".
    stub_bars.set("MSFT", o=400, h=401, l=399, c=400)
    from src.brokers.base import OrderRequest
    broker.place_order(OrderRequest(symbol="MSFT", side="buy", qty=5,
                                      sub_account="long", agent="manual"))
    agent = LongTermAgent(tmp_db, broker, FakeLongTerm(),
                           provider=ScriptedProvider([]), now=CLOCK)
    out = agent.run_once()
    assert out.status == "no-op"
