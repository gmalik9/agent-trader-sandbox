"""Shared agent policy: kill switch + decision validation.

Every agent calls `policy.check_kill_switch(conn)` before running and
`policy.validate(decision, broker, sub_account)` before each `place_order`.

Hard rules here are belt-and-suspenders on top of the broker's own caps so
that decisions are rejected *before* hitting the broker — this gives the
agent a chance to record a `reject_reason` in `agent_runs.decisions` and
move on without polluting the orders table with `rejected` rows from
obviously-bad LLM choices.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal

from src.brokers.base import BrokerBase
from src.brokers.sandbox_broker import DEFAULT_BLOCKLIST
from src.sandbox import db as dbm


Side = Literal["buy", "sell"]


@dataclass
class Decision:
    symbol: str
    side: Side
    qty: float
    order_type: str = "market"
    limit_price: float | None = None
    stop_price: float | None = None
    thesis: str | None = None
    # populated by policy.validate()
    accepted: bool = True
    reject_reason: str | None = None


def kill_switch_on(conn: sqlite3.Connection, agent: str | None = None) -> bool:
    """True if global or agent-specific kill switch is set in `settings`."""
    if dbm.get_setting(conn, "kill_switch") == "on":
        return True
    if agent and dbm.get_setting(conn, f"kill_switch.{agent}") == "on":
        return True
    return False


def validate(decision: Decision, broker: BrokerBase, sub_account: str,
             *, max_symbol_pct: float = 25.0, max_order_usd: float = 25_000.0,
             blocklist: set[str] | None = None) -> Decision:
    sym = decision.symbol.upper()
    blocklist = blocklist if blocklist is not None else DEFAULT_BLOCKLIST

    if decision.qty <= 0:
        decision.accepted = False
        decision.reject_reason = "qty<=0"
        return decision

    if sym in blocklist:
        decision.accepted = False
        decision.reject_reason = "blocklist"
        return decision

    # Notional cap — needs a reference price.
    ref = decision.limit_price
    if ref is None:
        positions = {p.symbol: p for p in broker.list_positions(sub_account)}
        ref = positions[sym].mark_price if sym in positions else None
    if ref is not None:
        notional = ref * decision.qty
        if notional > max_order_usd:
            decision.accepted = False
            decision.reject_reason = f"max_order_usd:{max_order_usd}"
            return decision
        acct = broker.get_account(sub_account)
        if acct.equity > 0 and decision.side == "buy":
            if 100.0 * notional / acct.equity > max_symbol_pct:
                decision.accepted = False
                decision.reject_reason = f"max_symbol_pct:{max_symbol_pct}"
                return decision

    # No-shorting at the policy layer.
    if decision.side == "sell":
        positions = {p.symbol: p.qty for p in broker.list_positions(sub_account)}
        if positions.get(sym, 0.0) + 1e-9 < decision.qty:
            decision.accepted = False
            decision.reject_reason = "no_shorting"
            return decision

    return decision
