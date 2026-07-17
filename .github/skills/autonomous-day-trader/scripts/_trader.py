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

# Running a script by path puts the script's dir on sys.path, not the app root —
# add the repo root (4 levels up) so `import src.*` works.
_APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)


def build_agent():
    """Return (agent, clients) — a DayTraderAgent with a live Alpaca-paper broker
    and started data clients, but NO LLM provider. Caller must `close(clients)`.
    """
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
    return agent, (short_term, long_term)


def close(clients) -> None:
    for c in clients:
        try:
            c.stop()
        except Exception:
            pass
