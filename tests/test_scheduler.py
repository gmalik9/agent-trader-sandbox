"""Scheduler smoke tests — verify jobs wire up and `tick_poll` drains requests."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.scheduler.runner import SchedulerRunner
from src.sandbox import db as dbm


@pytest.fixture
def runner(tmp_path, monkeypatch):
    # Point config to a temp DB and skip MCP/LLM setup.
    monkeypatch.setattr("src.scheduler.runner.db_path", lambda: tmp_path / "test.sqlite")
    monkeypatch.setattr("src.scheduler.runner.DATA_DIR", tmp_path)
    monkeypatch.setattr("src.scheduler.runner.LOCK_PATH", tmp_path / "scheduler.lock")
    r = SchedulerRunner()
    # Manually populate what setup() would, without spawning MCP subprocesses.
    r.conn = dbm.get_conn(tmp_path / "test.sqlite")
    dbm.migrate(r.conn)
    dbm.bootstrap_accounts(r.conn, capital_total=100_000.0, split_day_pct=30.0)
    r.broker = MagicMock()
    yield r
    r.conn.close()


def test_register_jobs_attaches_expected_ids(runner):
    runner.register_jobs()
    job_ids = {j.id for j in runner.scheduler.get_jobs()}
    assert job_ids == {"mtm", "reconcile", "day_tick", "scan_refresh",
                        "stop_monitor", "long_tick", "coord_tick", "tick_poll"}


def test_tick_poll_consumes_request_and_dispatches(runner, monkeypatch):
    # Patch the agent factories so we don't need an LLM provider.
    fake_day = MagicMock()
    fake_long = MagicMock()
    fake_coord = MagicMock()
    monkeypatch.setattr(runner, "_day_agent", lambda: fake_day)
    monkeypatch.setattr(runner, "_long_agent", lambda: fake_long)
    monkeypatch.setattr(runner, "_coordinator", lambda: fake_coord)

    runner.conn.execute(
        "INSERT INTO tick_requests(ts, agent, requested_by) VALUES (?, 'day', 'test')",
        (datetime.now(timezone.utc).isoformat(),),
    )
    runner.conn.execute(
        "INSERT INTO tick_requests(ts, agent, requested_by) VALUES (?, 'long', 'test')",
        (datetime.now(timezone.utc).isoformat(),),
    )

    runner.job_tick_poll()

    fake_day.run_once.assert_called_once()
    fake_long.run_once.assert_called_once()

    # Both requests should now be marked consumed.
    rows = runner.conn.execute(
        "SELECT consumed_at FROM tick_requests ORDER BY id"
    ).fetchall()
    assert all(r["consumed_at"] is not None for r in rows)


def test_tick_poll_coalesces_duplicate_agent_requests(runner, monkeypatch):
    """Multiple pending requests for the same agent run it ONCE and consume all."""
    fake_day = MagicMock()
    monkeypatch.setattr(runner, "_day_agent", lambda: fake_day)
    # Five rapid "Tick day now" clicks.
    for _ in range(5):
        runner.conn.execute(
            "INSERT INTO tick_requests(ts, agent, requested_by) VALUES (?, 'day', 'test')",
            (datetime.now(timezone.utc).isoformat(),),
        )
    runner.job_tick_poll()
    # The agent ran exactly once despite 5 queued requests…
    fake_day.run_once.assert_called_once()
    # …and every one of the 5 requests is marked consumed.
    rows = runner.conn.execute("SELECT consumed_at FROM tick_requests").fetchall()
    assert len(rows) == 5 and all(r["consumed_at"] is not None for r in rows)


def test_tick_poll_records_consumption_even_when_agent_raises(runner, monkeypatch):
    bomb = MagicMock()
    bomb.run_once.side_effect = RuntimeError("boom")
    monkeypatch.setattr(runner, "_day_agent", lambda: bomb)

    runner.conn.execute(
        "INSERT INTO tick_requests(ts, agent, requested_by) VALUES (?, 'day', 'test')",
        (datetime.now(timezone.utc).isoformat(),),
    )
    runner.job_tick_poll()
    row = runner.conn.execute("SELECT consumed_at FROM tick_requests").fetchone()
    assert row["consumed_at"] is not None


def test_scan_refresh_triggers_scan_run(runner, monkeypatch):
    """The scan_refresh job must call scan_run to unfreeze the idea cache."""
    monkeypatch.setattr("src.scheduler.runner.is_market_open", lambda _now: True)
    st = MagicMock()
    runner.short_term = st
    runner.job_scan_refresh()
    st.scan_run.assert_called_once()
    assert st.scan_run.call_args.kwargs.get("mode") == "intraday"
    # Uses the configured universe (default sp500 for a broad candidate pool).
    assert st.scan_run.call_args.kwargs.get("universe") == runner.settings.scan_universe


def test_scan_refresh_swallows_timeout(runner, monkeypatch):
    """A client-side timeout is benign (server keeps scanning) — must not raise."""
    monkeypatch.setattr("src.scheduler.runner.is_market_open", lambda _now: True)
    st = MagicMock()
    st.scan_run.side_effect = TimeoutError()
    runner.short_term = st
    runner.job_scan_refresh()  # should not raise


def test_scan_refresh_skips_when_market_closed(runner, monkeypatch):
    monkeypatch.setattr("src.scheduler.runner.is_market_open", lambda _now: False)
    st = MagicMock()
    runner.short_term = st
    runner.job_scan_refresh()
    st.scan_run.assert_not_called()


def test_stop_monitor_runs_agent_manager(runner, monkeypatch):
    """The stop_monitor job invokes the day agent's LLM-free position manager."""
    monkeypatch.setattr("src.scheduler.runner.is_market_open", lambda _now: True)
    fake_day = MagicMock()
    monkeypatch.setattr(runner, "_day_agent", lambda: fake_day)
    runner.job_stop_monitor()
    fake_day.manage_positions_only.assert_called_once()


def test_stop_monitor_skips_when_market_closed(runner, monkeypatch):
    monkeypatch.setattr("src.scheduler.runner.is_market_open", lambda _now: False)
    fake_day = MagicMock()
    monkeypatch.setattr(runner, "_day_agent", lambda: fake_day)
    runner.job_stop_monitor()
    fake_day.manage_positions_only.assert_not_called()
