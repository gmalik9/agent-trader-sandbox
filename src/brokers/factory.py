"""Factory for the configured broker.

Used by the scheduler and the Streamlit "tick now" path so both processes
agree on which venue(s) are receiving orders.
"""

from __future__ import annotations

import logging
import sqlite3

from src.brokers.alpaca_paper_broker import AlpacaPaperBroker
from src.brokers.base import BrokerBase
from src.brokers.dual_broker import DualBroker
from src.brokers.sandbox_broker import SandboxBroker
from src.config import get_settings
from src.mcp_clients.long_term import LongTermClient

log = logging.getLogger(__name__)


def build_broker(conn: sqlite3.Connection,
                 long_term_client: LongTermClient | None = None) -> BrokerBase:
    s = get_settings()
    backend = (s.broker_backend or "sandbox").lower()

    if backend == "sandbox":
        return SandboxBroker(conn)

    if backend == "alpaca_paper":
        if long_term_client is None:
            raise RuntimeError("alpaca_paper backend requires a LongTermClient")
        return AlpacaPaperBroker(conn, long_term_client)

    if backend == "dual":
        if long_term_client is None:
            raise RuntimeError("dual backend requires a LongTermClient")
        primary = SandboxBroker(conn)
        secondary = AlpacaPaperBroker(conn, long_term_client)
        return DualBroker(conn, primary=primary, secondary=secondary)

    raise ValueError(f"unknown BROKER_BACKEND: {backend!r}")
