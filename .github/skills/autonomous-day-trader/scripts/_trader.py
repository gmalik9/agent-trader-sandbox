#!/usr/bin/env python3
"""Shared bootstrap for the Copilot-driven trading scripts.

Builds the exact same broker + data clients the scheduler uses, but with NO LLM
provider (`provider=None`) — the reasoning is done by Copilot, not an API call.
Run inside the scheduler container so the sibling MCP subprocesses launch:

    docker compose exec -T scheduler python .../scripts/gather_context.py
    docker compose exec -T scheduler python .../scripts/execute_trade.py ...
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

# Running a script by path puts the script's dir on sys.path, not the app root —
# add the repo root (4 levels up) so `import src.*` works.
_APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

# The autonomous-day-trader skill trades a DEDICATED Alpaca paper account,
# separate from the scheduler agent's account, so the two never touch the same
# book. Credentials come from COPILOT_SKILLS_ALPACA_* in secrets.toml; the
# account number is asserted after connecting (see `_assert_account`).
EXPECTED_ACCOUNT_DEFAULT = "PA3WWRZG806F"


def _bind_skill_account() -> str:
    """Point THIS process (the skill only) at the Copilot-skills paper account.

    Overrides ``ALPACA_*`` in ``os.environ`` from the ``COPILOT_SKILLS_ALPACA_*``
    keys in secrets.toml so both the broker MCP subprocess (which reads
    ``os.environ`` at spawn) and the repo-side Alpaca REST leg (via
    ``get_settings()``) trade the skill's dedicated account. Because each skill
    script runs as its own ``docker compose exec`` process, this NEVER affects
    the long-running scheduler agent, which keeps its own env / account.

    Returns the expected account number for the post-connect safety assertion.
    """
    import tomllib
    from pathlib import Path

    secrets: dict[str, object] = {}
    sp = Path(_APP_ROOT) / ".streamlit" / "secrets.toml"
    if sp.exists():
        try:
            with sp.open("rb") as fh:
                secrets = {str(k): v for k, v in tomllib.load(fh).items()
                           if not isinstance(v, dict)}
        except Exception:  # noqa: BLE001 — a bad secrets file must yield a clear exit
            secrets = {}

    def _pick(name: str) -> str:
        return (os.environ.get(name) or str(secrets.get(name, ""))).strip()

    key = _pick("COPILOT_SKILLS_ALPACA_API_KEY_ID")
    secret = _pick("COPILOT_SKILLS_ALPACA_SECRET_KEY")
    expected = _pick("COPILOT_SKILLS_ALPACA_ACCOUNT") or EXPECTED_ACCOUNT_DEFAULT

    if not key or not secret:
        sys.exit(
            "FATAL: COPILOT_SKILLS_ALPACA_API_KEY_ID / "
            "COPILOT_SKILLS_ALPACA_SECRET_KEY are not set in "
            ".streamlit/secrets.toml. The autonomous-day-trader skill refuses to "
            f"trade without the dedicated paper account ({expected}) keys, so it "
            "can never accidentally trade the scheduler agent's account."
        )

    os.environ["ALPACA_API_KEY_ID"] = key
    os.environ["ALPACA_SECRET_KEY"] = secret
    os.environ["ALPACA_PAPER"] = "true"

    # get_settings() is lru_cached and may already hold the agent's keys; bust it
    # so the repo-side Alpaca REST leg rebuilds Settings from the override above.
    try:
        from src.config import get_settings
        get_settings.cache_clear()
    except Exception:  # noqa: BLE001
        pass

    return expected


def _assert_account(long_term, expected: str) -> None:
    """Hard safety gate: abort if the connected Alpaca account isn't the skill's.

    Runs before any broker is built, so a mis-keyed credential can never place an
    order on the wrong (e.g. the scheduler agent's) account.
    """
    try:
        acct = long_term.get_account() or {}
    except Exception as e:  # noqa: BLE001
        sys.exit(f"FATAL: could not read the Alpaca account to verify identity: {e}")
    num = str(acct.get("account_number") or "")
    if not num.startswith("PA"):
        sys.exit(
            f"FATAL: account_number {num!r} is not a paper account (expected "
            "a 'PA...' number). Refusing to trade."
        )
    if num != expected:
        sys.exit(
            f"SAFETY ABORT: connected Alpaca account {num!r} != the expected "
            f"skill account {expected!r}. Refusing to trade the wrong account — "
            "check COPILOT_SKILLS_ALPACA_* in .streamlit/secrets.toml."
        )


def build_agent():
    """Return (agent, clients) — a DayTraderAgent with a live Alpaca-paper broker
    and started data clients, but NO LLM provider. Caller must `close(clients)`.

    The broker is bound to the skill's dedicated paper account and the account
    identity is asserted before any order can be placed.
    """
    expected = _bind_skill_account()

    from src.agents.day_trader import DayTraderAgent
    from src.brokers.factory import build_broker
    from src.config import db_path
    from src.mcp_clients.long_term import LongTermClient
    from src.mcp_clients.short_term import ShortTermClient
    from src.sandbox import db as dbm

    conn = dbm.get_conn(db_path())
    dbm.migrate(conn)

    short_term = ShortTermClient()
    short_term.start()
    long_term = LongTermClient()
    long_term.start()

    # Hard gate: refuse to proceed unless we're on the skill's paper account.
    _assert_account(long_term, expected)

    broker = build_broker(conn, long_term_client=long_term)

    options = None
    try:
        from src.brokers.alpaca_options import AlpacaOptions
        o = AlpacaOptions()
        if o.options_enabled():
            options = o
    except Exception:
        options = None

    agent = DayTraderAgent(conn, broker, short_term, provider=None,
                           options=options, long_term=long_term)
    # Keep the skill's DB state isolated from the standalone scheduler agent.
    # Both can trade different Alpaca accounts, but if they share sub_account
    # ('day') they race on stop-plan rows and can prune each other's plans.
    agent.sub_account = os.environ.get("COPILOT_SKILLS_SUB_ACCOUNT", "copilot_skills")
    # First run may not have this account row yet; create it once.
    row = conn.execute("SELECT id FROM accounts WHERE name=?", (agent.sub_account,)).fetchone()
    if row is None:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO accounts(name, venue, starting_cash, created_at) VALUES (?, ?, ?, ?)",
            (agent.sub_account, "alpaca_paper", 0.0, now),
        )
    return agent, (short_term, long_term)


def close(clients) -> None:
    for c in clients:
        try:
            c.stop()
        except Exception:
            pass
