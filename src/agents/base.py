"""Common agent base: persists `agent_runs` rows, owns LLM tool loop wiring."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.agents.policy import Decision, kill_switch_on
from src.brokers.base import BrokerBase
from src.llm.provider import LLMProvider
from src.llm.tool_loop import LoopResult, ToolHandler, run_tool_loop
from src.sandbox import db as dbm

log = logging.getLogger(__name__)


@dataclass
class RunOutcome:
    status: str               # ok | halted | error | no-op
    decisions: list[dict] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)
    error: str | None = None
    latency_ms: int = 0
    run_id: int | None = None


class AgentBase:
    name: str  # 'day' | 'long' | 'coordinator'
    sub_account: str

    def __init__(self, conn: sqlite3.Connection, broker: BrokerBase,
                 provider: LLMProvider | None = None) -> None:
        self.conn = conn
        self.broker = broker
        self.provider = provider

    # ----- helpers -----

    def _account_id(self) -> int:
        return dbm.get_account_id(self.conn, self.sub_account)

    def _record_run(self, *, status: str, prompt: str, response: str | None,
                    tools_called: Any, decisions: Any, error: str | None,
                    latency_ms: int) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO agent_runs(account_id, ts, agent, status, prompt, response,
                                   tools_called, decisions, error, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (self._account_id(), datetime.now(timezone.utc).isoformat(),
             self.name, status, prompt, response,
             json.dumps(tools_called, default=str) if tools_called is not None else None,
             json.dumps(decisions, default=str) if decisions is not None else None,
             error, latency_ms),
        )
        rid = cur.lastrowid
        self._append_reasoning_log(rid, status, response, decisions, tools_called, error)
        return rid

    def _append_reasoning_log(self, run_id: int, status: str, response: str | None,
                              decisions: Any, tools_called: Any, error: str | None) -> None:
        """Durably append this run's reasoning to data/reasoning_log.jsonl."""
        try:
            from src.analysis import reasoning as R
            from src.config import DATA_DIR
            R.append_run(
                run_id=run_id, ts=datetime.now(timezone.utc).isoformat(),
                agent=self.name, status=status, response=response,
                decisions_obj=decisions, tools_called_obj=tools_called, error=error,
                path=DATA_DIR / "reasoning_log.jsonl",
            )
        except Exception:  # noqa: BLE001 — logging must never break a run
            log.debug("reasoning append failed", exc_info=True)

    def _kill_switched(self) -> bool:
        return kill_switch_on(self.conn, agent=self.name)

    def _run_llm(self, system: str, user: str, handlers: list[ToolHandler],
                 *, max_steps: int = 8, deadline_seconds: float | None = None) -> LoopResult:
        if self.provider is None:
            raise RuntimeError(f"{self.name}: no LLM provider configured")
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return run_tool_loop(self.provider, msgs, handlers, max_steps=max_steps,
                             deadline_seconds=deadline_seconds)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def time_ms() -> int:
    return int(time.time() * 1000)
