"""Day-Trader agent.

Workflow per `run_once()`:
  1. Kill-switch + market-window gate.
  2. Snapshot starting equity (for the -2% intraday drawdown halt).
  3. Force-flat if past 15:55 ET → close all open positions, cancel pending.
  4. Else: hand the LLM a curated tool set, let it call `propose_trade(...)` 0+
     times via a tool-loop. We collect those proposals.
  5. Validate each via `policy.validate(...)` + apply hard rules
     (max 5 concurrent positions, 1% account risk via ATR stop).
  6. Place the surviving orders. Record everything in `agent_runs`.

The LLM never reaches the broker directly — proposals are buffered and only
the agent code calls `broker.place_order(...)`.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, time, timezone
from typing import Any

import pandas as pd

from src.agents.base import AgentBase, RunOutcome, now_utc, time_ms
from src.agents.policy import Decision, validate
from src.brokers.base import BrokerBase, OrderRequest
from src.llm.provider import LLMProvider, ToolSpec
from src.llm.tool_loop import ToolHandler
from src.mcp_clients.short_term import ShortTermClient
from src.sandbox import db as dbm
from src.sandbox.clock import is_force_flat_window, is_market_open

log = logging.getLogger(__name__)

MAX_CONCURRENT_POSITIONS = 5
ACCOUNT_RISK_PCT = 0.01           # 1% of equity per trade
DD_HALT_PCT = -2.0                # halt for the day at -2% intraday
DEFAULT_ATR_PCT = 0.02            # 2% fallback ATR when upstream doesn't return one

SYSTEM_PROMPT = """You are a disciplined intraday equities trader running on a paper account.
You see a curated list of intraday ideas and per-symbol quotes. Your job:
- Identify at most 3 high-conviction long entries from the ideas list.
- Call `propose_trade` once per intended entry with: symbol, entry_price, stop_price, thesis.
- Then output a brief one-paragraph rationale.
Constraints (you do NOT need to enforce these — the runtime will):
- The runtime sizes positions for 1% account risk via your stop.
- Max 5 concurrent positions across the book.
- No shorting, no leveraged/inverse ETFs.
- All positions are force-closed by 15:55 ET, so do not open new positions after 15:30 ET.
If no idea has both a clear setup and a defined stop, output `no trades today` and call no tools.
"""


@dataclass
class _Proposal:
    symbol: str
    entry_price: float
    stop_price: float
    thesis: str = ""


class DayTraderAgent(AgentBase):
    name = "day"
    sub_account = "day"

    def __init__(self, conn: sqlite3.Connection, broker: BrokerBase,
                 short_term: ShortTermClient, provider: LLMProvider | None = None,
                 *, now: datetime | None = None) -> None:
        super().__init__(conn, broker, provider)
        self.short_term = short_term
        self._now = now  # injectable for tests

    def _wall(self) -> datetime:
        return self._now or now_utc()

    # ---------------- main entrypoint ----------------

    def run_once(self) -> RunOutcome:
        start = time_ms()
        if self._kill_switched():
            rid = self._record_run(status="halted", prompt="", response=None,
                                    tools_called=None, decisions=None,
                                    error="kill_switch", latency_ms=time_ms() - start)
            return RunOutcome(status="halted", error="kill_switch", run_id=rid)

        wall = self._wall()

        # Snapshot starting equity for the day for the DD halt check.
        starting_equity = self._starting_equity_today(wall)
        acct = self.broker.get_account(self.sub_account)
        if starting_equity > 0:
            dd_pct = 100.0 * (acct.equity - starting_equity) / starting_equity
            if dd_pct <= DD_HALT_PCT:
                rid = self._record_run(status="halted", prompt="", response=None,
                                        tools_called=None,
                                        decisions=[{"halt": "daily_drawdown",
                                                    "starting_equity": starting_equity,
                                                    "equity": acct.equity, "dd_pct": dd_pct}],
                                        error=f"drawdown:{dd_pct:.2f}%",
                                        latency_ms=time_ms() - start)
                return RunOutcome(status="halted", error="daily_drawdown", run_id=rid)

        # Force-flat window: close everything, no new trades.
        if is_force_flat_window(wall):
            decisions, orders = self._force_flat()
            rid = self._record_run(status="ok", prompt="force_flat", response=None,
                                    tools_called=None, decisions=decisions,
                                    error=None, latency_ms=time_ms() - start)
            return RunOutcome(status="ok", decisions=decisions, orders=orders, run_id=rid)

        # Don't trade outside of market hours.
        if not is_market_open(wall):
            rid = self._record_run(status="no-op", prompt="market_closed", response=None,
                                    tools_called=None, decisions=None, error=None,
                                    latency_ms=time_ms() - start)
            return RunOutcome(status="no-op", run_id=rid)

        # Ask the LLM.
        proposals, loop_res = self._run_llm_loop()
        decisions = self._validate_and_size(proposals)
        orders = self._place(decisions)

        rid = self._record_run(
            status="ok", prompt=SYSTEM_PROMPT[:200], response=loop_res.final_text,
            tools_called=[asdict_step(s) for s in loop_res.steps],
            decisions=[d.__dict__ for d in decisions],
            error=None, latency_ms=time_ms() - start,
        )
        return RunOutcome(status="ok", decisions=[d.__dict__ for d in decisions],
                          orders=orders, run_id=rid)

    # ---------------- pieces ----------------

    def _starting_equity_today(self, wall: datetime) -> float:
        aid = self._account_id()
        start_of_day = wall.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        row = self.conn.execute(
            "SELECT equity FROM equity_curve WHERE account_id=? AND ts >= ? ORDER BY ts LIMIT 1",
            (aid, start_of_day),
        ).fetchone()
        if row:
            return float(row["equity"])
        return self.broker.get_account(self.sub_account).equity

    def _force_flat(self) -> tuple[list[dict], list[dict]]:
        positions = self.broker.list_positions(self.sub_account)
        decisions: list[dict] = []
        orders: list[dict] = []
        for p in positions:
            if p.qty <= 0:
                continue
            decisions.append({"symbol": p.symbol, "side": "sell", "qty": p.qty,
                               "reason": "force_flat"})
            res = self.broker.close_position(p.symbol, sub_account=self.sub_account,
                                              percentage=100.0)
            orders.append({"id": res.id, "symbol": p.symbol, "status": res.status,
                            "fill_price": res.fill_price})
        return decisions, orders

    def _run_llm_loop(self):
        proposals: list[_Proposal] = []

        def list_intraday_ideas(*, tier: str = "A", limit: int = 10) -> dict:
            try:
                return self.short_term.list_ideas(mode="intraday", tier=tier, limit=limit)
            except Exception as e:
                return {"error": str(e)}

        def get_quote(symbol: str) -> dict:
            try:
                return self.short_term.lookup_ticker(symbol, interval="5m", period="1d")
            except Exception as e:
                return {"error": str(e)}

        def current_positions() -> list[dict]:
            return [p.__dict__ for p in self.broker.list_positions(self.sub_account)]

        def account_snapshot() -> dict:
            return self.broker.get_account(self.sub_account).__dict__

        def propose_trade(*, symbol: str, entry_price: float, stop_price: float,
                          thesis: str = "") -> dict:
            proposals.append(_Proposal(symbol=symbol.upper(),
                                         entry_price=float(entry_price),
                                         stop_price=float(stop_price),
                                         thesis=thesis))
            return {"ok": True, "buffered": len(proposals)}

        handlers = [
            ToolHandler(spec=ToolSpec(
                name="list_intraday_ideas",
                description="List intraday trade ideas from the upstream scanner.",
                json_schema={"type": "object", "properties": {
                    "tier": {"type": "string", "enum": ["A", "B", "C"], "default": "A"},
                    "limit": {"type": "integer", "default": 10},
                }}), fn=list_intraday_ideas),
            ToolHandler(spec=ToolSpec(
                name="get_quote",
                description="Get a recent 5m quote/series for a symbol.",
                json_schema={"type": "object", "required": ["symbol"], "properties": {
                    "symbol": {"type": "string"}}}), fn=get_quote),
            ToolHandler(spec=ToolSpec(
                name="current_positions", description="List currently held day-trading positions.",
                json_schema={"type": "object"}), fn=current_positions),
            ToolHandler(spec=ToolSpec(
                name="account_snapshot", description="Equity, cash, positions value.",
                json_schema={"type": "object"}), fn=account_snapshot),
            ToolHandler(spec=ToolSpec(
                name="propose_trade",
                description=("Buffer a long entry proposal. Runtime will size it for "
                              "1% account risk based on (entry - stop) and place the order."),
                json_schema={"type": "object",
                              "required": ["symbol", "entry_price", "stop_price"],
                              "properties": {
                                  "symbol": {"type": "string"},
                                  "entry_price": {"type": "number"},
                                  "stop_price": {"type": "number"},
                                  "thesis": {"type": "string"}}}),
                fn=propose_trade),
        ]

        user = (f"It is {self._wall().isoformat()}. You have a day-trading sub-account. "
                "Use the tools to evaluate today's ideas, then call `propose_trade` for "
                "any high-conviction longs.")
        loop_res = self._run_llm(SYSTEM_PROMPT, user, handlers, max_steps=8)
        return proposals, loop_res

    def _validate_and_size(self, proposals: list[_Proposal]) -> list[Decision]:
        acct = self.broker.get_account(self.sub_account)
        open_syms = {p.symbol for p in self.broker.list_positions(self.sub_account) if p.qty > 0}
        out: list[Decision] = []
        slots_left = MAX_CONCURRENT_POSITIONS - len(open_syms)

        for p in proposals:
            d = Decision(symbol=p.symbol, side="buy", qty=0.0, thesis=p.thesis)
            if slots_left <= 0 and p.symbol not in open_syms:
                d.accepted = False
                d.reject_reason = "max_concurrent_positions"
                out.append(d)
                continue
            risk_per_share = max(p.entry_price - p.stop_price, 1e-6)
            if risk_per_share >= p.entry_price:
                d.accepted = False
                d.reject_reason = "stop_below_zero_or_inverted"
                out.append(d)
                continue
            risk_budget = acct.equity * ACCOUNT_RISK_PCT
            qty = math.floor(risk_budget / risk_per_share)
            if qty <= 0:
                d.accepted = False
                d.reject_reason = "size_rounded_to_zero"
                out.append(d)
                continue
            d.qty = float(qty)
            d = validate(d, self.broker, self.sub_account)
            if d.accepted and p.symbol not in open_syms:
                slots_left -= 1
            out.append(d)
        return out

    def _place(self, decisions: list[Decision]) -> list[dict]:
        orders: list[dict] = []
        for d in decisions:
            if not d.accepted:
                continue
            res = self.broker.place_order(OrderRequest(
                symbol=d.symbol, side=d.side, qty=d.qty, order_type=d.order_type,
                limit_price=d.limit_price, sub_account=self.sub_account,
                agent=self.name, thesis=d.thesis,
            ))
            orders.append({"id": res.id, "symbol": d.symbol, "status": res.status,
                            "fill_price": res.fill_price})
        return orders


def asdict_step(s) -> dict:
    return {"step": s.step, "text": s.text, "tool_calls": s.tool_calls}
