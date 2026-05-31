from __future__ import annotations

from datetime import datetime, timezone

from src.agents.coordinator import Coordinator
from src.brokers.sandbox_broker import SandboxBroker
from src.sandbox import db as dbm


CLOCK = datetime(2025, 2, 3, 21, 45, tzinfo=timezone.utc)


def test_no_op_when_balanced(tmp_db, stub_bars):
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    coord = Coordinator(tmp_db, broker, now=CLOCK)
    out = coord.run_once()
    assert out.status == "no-op"


def test_transfers_cash_when_drifted(tmp_db, stub_bars, monkeypatch):
    monkeypatch.setenv("SLIPPAGE_BPS", "0")
    monkeypatch.setenv("COMMISSION_BPS", "0")
    from src import config
    config.get_settings.cache_clear()

    # Skew cash directly: move $5k from day → long via paired ledger entries,
    # so net cash flow is zero (coordinator requires balanced net).
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    day_aid = dbm.get_account_id(tmp_db, "day")
    long_aid = dbm.get_account_id(tmp_db, "long")
    tmp_db.execute(
        "INSERT INTO cash_ledger(account_id, ts, delta, reason) VALUES (?, ?, ?, 'fee')",
        (day_aid, CLOCK.isoformat(), -5_000.0),
    )
    tmp_db.execute(
        "INSERT INTO cash_ledger(account_id, ts, delta, reason) VALUES (?, ?, ?, 'deposit')",
        (long_aid, CLOCK.isoformat(), 5_000.0),
    )
    # Pre-state: day=$25k (target $30k), long=$75k (target $70k). Net = 0.

    coord = Coordinator(tmp_db, broker, now=CLOCK)
    out = coord.run_once()
    assert out.status == "ok", out.error
    day_cash = dbm.get_cash(tmp_db, day_aid)
    long_cash = dbm.get_cash(tmp_db, long_aid)
    assert abs(day_cash - 30_000.0) < 1.0
    assert abs(long_cash - 70_000.0) < 1.0


def test_kill_switch_halts_transfers(tmp_db, stub_bars):
    dbm.set_setting(tmp_db, "kill_switch", "on")
    broker = SandboxBroker(tmp_db, bar_provider=stub_bars)
    out = Coordinator(tmp_db, broker, now=CLOCK).run_once()
    assert out.status == "halted"
