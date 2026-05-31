"""Coordinator — monthly cash rebalance between day and long sub-accounts.

No LLM. Pure code.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone

from src.agents.base import AgentBase, RunOutcome, now_utc, time_ms
from src.agents.policy import kill_switch_on
from src.brokers.base import BrokerBase
from src.config import get_settings
from src.sandbox import db as dbm

log = logging.getLogger(__name__)


class Coordinator(AgentBase):
    name = "coordinator"
    sub_account = "master"   # ledger root for transfers

    def __init__(self, conn: sqlite3.Connection, broker: BrokerBase,
                 *, now: datetime | None = None) -> None:
        super().__init__(conn, broker, provider=None)
        self._now = now

    def _wall(self) -> datetime:
        return self._now or now_utc()

    def run_once(self) -> RunOutcome:
        start = time_ms()
        # Either-agent kill-switch blocks transfers.
        if (kill_switch_on(self.conn) or kill_switch_on(self.conn, agent="day")
                or kill_switch_on(self.conn, agent="long")):
            rid = self._record_run(status="halted", prompt="", response=None,
                                    tools_called=None, decisions=None,
                                    error="kill_switch", latency_ms=time_ms() - start)
            return RunOutcome(status="halted", error="kill_switch", run_id=rid)

        s = get_settings()
        capital_total = s.capital_total
        split_day_pct = s.split_day_pct
        target_day = capital_total * split_day_pct / 100.0
        target_long = capital_total - target_day

        day_acct = self.broker.get_account("day")
        long_acct = self.broker.get_account("long")

        delta_day = target_day - day_acct.cash
        delta_long = target_long - long_acct.cash

        # Only transfer when both sides agree on a net flow that conserves the master.
        net = delta_day + delta_long
        if abs(net) > 1.0:
            rid = self._record_run(
                status="error", prompt="", response=None, tools_called=None,
                decisions=[{"target_day": target_day, "target_long": target_long,
                             "day_cash": day_acct.cash, "long_cash": long_acct.cash}],
                error=f"unbalanced_net:{net:.2f}",
                latency_ms=time_ms() - start,
            )
            return RunOutcome(status="error", error="unbalanced_net", run_id=rid)

        # Only act if the imbalance is meaningful (>$50).
        if abs(delta_day) < 50.0:
            rid = self._record_run(status="no-op", prompt="balanced", response=None,
                                    tools_called=None, decisions=None, error=None,
                                    latency_ms=time_ms() - start)
            return RunOutcome(status="no-op", run_id=rid)

        ts = self._wall().isoformat()
        day_aid = dbm.get_account_id(self.conn, "day")
        long_aid = dbm.get_account_id(self.conn, "long")
        # Two paired ledger entries; the master account stays untouched because
        # cash flows directly between the two sub-accounts.
        self.conn.execute(
            "INSERT INTO cash_ledger(account_id, ts, delta, reason) VALUES (?, ?, ?, 'transfer')",
            (day_aid, ts, delta_day),
        )
        self.conn.execute(
            "INSERT INTO cash_ledger(account_id, ts, delta, reason) VALUES (?, ?, ?, 'transfer')",
            (long_aid, ts, delta_long),
        )
        decisions = [{"transfer_day": delta_day, "transfer_long": delta_long}]
        rid = self._record_run(status="ok", prompt="", response=None,
                                tools_called=None, decisions=decisions, error=None,
                                latency_ms=time_ms() - start)
        return RunOutcome(status="ok", decisions=decisions, run_id=rid)
