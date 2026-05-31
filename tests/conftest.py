"""Shared pytest fixtures."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.sandbox import db as dbm
from src.sandbox.engine import Bar, BarProvider


@pytest.fixture
def tmp_db(tmp_path: Path) -> sqlite3.Connection:
    p = tmp_path / "test.sqlite"
    conn = dbm.get_conn(p)
    dbm.migrate(conn)
    dbm.bootstrap_accounts(conn, capital_total=100_000.0, split_day_pct=30.0)
    yield conn
    conn.close()


class StubBarProvider:
    """In-memory bar provider for tests."""

    def __init__(self) -> None:
        self.bars: dict[str, Bar] = {}

    def set(self, symbol: str, *, o: float, h: float, l: float, c: float,
            v: float = 1_000_000.0) -> None:
        self.bars[symbol.upper()] = Bar(
            ts=datetime.now(timezone.utc),
            open=o, high=h, low=l, close=c, volume=v,
        )

    def get_bar(self, symbol: str, ts: datetime) -> Bar | None:
        return self.bars.get(symbol.upper())


@pytest.fixture
def stub_bars() -> StubBarProvider:
    return StubBarProvider()
