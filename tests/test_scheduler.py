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
    assert job_ids == {"mtm", "day_tick", "long_tick", "coord_tick", "tick_poll"}


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
