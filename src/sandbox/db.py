"""SQLite helpers — connection, migrations, account bootstrap."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.config import db_path, get_settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def get_conn(path: Path | None = None) -> sqlite3.Connection:
    p = path or db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    ddl = SCHEMA_PATH.read_text()
    conn.executescript(ddl)


def bootstrap_accounts(conn: sqlite3.Connection, *, capital_total: float | None = None,
                       split_day_pct: float | None = None) -> None:
    """Create the standard accounts (master/day/long/day_alpaca/long_alpaca) if missing.

    Cash for sandbox sub-accounts is deposited via cash_ledger so the
    SUM(delta) invariant is exact. Alpaca-mirror sub-accounts are created with
    zero starting cash (their truth is the remote Alpaca account).
    """
    s = get_settings()
    total = capital_total if capital_total is not None else s.capital_total
    day_pct = split_day_pct if split_day_pct is not None else s.split_day_pct
    day_cash = total * (day_pct / 100.0)
    long_cash = total - day_cash
    now = datetime.now(timezone.utc).isoformat()

    plan = [
        ("master", "ledger", total),
        ("day", "sandbox", day_cash),
        ("long", "sandbox", long_cash),
        ("day_alpaca", "alpaca_paper", 0.0),
        ("long_alpaca", "alpaca_paper", 0.0),
    ]
    for name, venue, starting in plan:
        row = conn.execute("SELECT id FROM accounts WHERE name = ?", (name,)).fetchone()
        if row is not None:
            continue
        cur = conn.execute(
            "INSERT INTO accounts(name, venue, starting_cash, created_at) VALUES (?, ?, ?, ?)",
            (name, venue, starting, now),
        )
        account_id = cur.lastrowid
        if venue == "sandbox" and starting > 0:
            conn.execute(
                "INSERT INTO cash_ledger(account_id, ts, delta, reason) VALUES (?, ?, ?, 'deposit')",
                (account_id, now, starting),
            )


def get_account_id(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute("SELECT id FROM accounts WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise KeyError(f"unknown account: {name}")
    return int(row["id"])


def get_cash(conn: sqlite3.Connection, account_id: int) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(delta), 0.0) AS c FROM cash_ledger WHERE account_id = ?",
        (account_id,),
    ).fetchone()
    return float(row["c"])


def get_setting(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO settings(key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, value, now),
    )


def upsert_position_plan(conn: sqlite3.Connection, *, account_id: int, symbol: str,
                         side: str, entry_price: float, stop_price: float,
                         target_price: float | None, agent: str) -> None:
    """Record (or replace) the active stop plan for a position.

    Any existing active plan for the same (account, symbol) is superseded so
    there is exactly one active plan per open position — a re-entry or size-up
    resets the stop to the latest one the agent set.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE position_plans SET active=0, closed_at=?, close_reason='superseded' "
        "WHERE account_id=? AND symbol=? AND active=1",
        (now, account_id, symbol.upper()),
    )
    conn.execute(
        "INSERT INTO position_plans(account_id, symbol, side, entry_price, stop_price, "
        "target_price, agent, active, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
        (account_id, symbol.upper(), side, float(entry_price), float(stop_price),
         (float(target_price) if target_price is not None else None), agent, now, now),
    )


def get_active_position_plans(conn: sqlite3.Connection, account_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM position_plans WHERE account_id=? AND active=1 ORDER BY symbol",
        (account_id,),
    ).fetchall()


def close_position_plan(conn: sqlite3.Connection, plan_id: int, reason: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE position_plans SET active=0, closed_at=?, close_reason=? WHERE id=?",
        (now, reason, plan_id),
    )
