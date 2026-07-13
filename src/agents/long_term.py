"""Long-Term Investor agent.

Runs at end-of-day on trading days. Workflow:
  1. Kill-switch gate.
  2. Compute current drift vs upstream `get_portfolio_suggestion`.
  3. If no rebalance happened in the last 7 days, OR drift on any symbol > 10%
     → run the rebalance flow.
  4. Hand the LLM `get_recommendations`, `lookup_ticker`, `get_news`,
     `current_positions`, and `propose_rebalance` (a buffer tool).
  5. Validate and place orders. Max 25% per symbol enforced at the policy layer.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

from src.agents.base import AgentBase, RunOutcome, now_utc, time_ms
from src.agents.day_trader import asdict_step
from src.agents.policy import Decision, validate
from src.brokers.base import BrokerBase, OrderRequest
from src.llm.provider import LLMProvider, ToolSpec
from src.llm.tool_loop import ToolHandler
from src.mcp_clients.long_term import LongTermClient
from src.sandbox import db as dbm

log = logging.getLogger(__name__)

REBALANCE_INTERVAL_DAYS = 7
DRIFT_OVERRIDE_PCT = 10.0
MAX_SYMBOL_PCT = 25.0

SYSTEM_PROMPT = """You are "Atlas-Long", a long-horizon portfolio manager running a paper account.
Your objective is to COMPOUND the account's value over months and years by owning
high-quality businesses at sensible weights, while controlling drawdowns. You act
infrequently and deliberately — most reviews should end in few or no changes.

════════ DECISION FRAMEWORK — weigh ALL of these ════════
1. QUALITY & FUNDAMENTALS. Prefer durable, profitable, growing businesses with a
   real competitive advantage. Use `get_recommendations` (upstream research),
   `lookup_ticker` (fundamentals/quote) and `get_news` (recent developments +
   sentiment) to build a view of each name.
2. VALUATION & CONVICTION. Concentrate in your highest-conviction names at larger
   weights; trim or avoid the overvalued or fundamentally deteriorating ones.
3. DIVERSIFICATION & RISK. Spread exposure across sectors; no single position
   should dominate the book. You MAY use broad leveraged ETFs (SSO, QLD) to add
   measured beta, or inverse / hedge ETFs (SH, PSQ, SQQQ) to hedge when the macro
   backdrop or a specific risk warrants it.
4. PORTFOLIO DRIFT. Compare current weights vs. targets and vs. the latest
   research. Rebalance ONLY when the drift or a genuine thesis change is material —
   avoid churn and unnecessary costs.

════════ HOW TO ACT ════════
- Call `propose_rebalance` once per intended change with: symbol, side
  (buy = raise toward target / sell = trim or close), target_weight_pct
  (0-100 of sub-account equity), and a thesis citing the fundamental case AND the
  news / sentiment backdrop.
- Make small, incremental moves toward targets rather than large abrupt swings.

════════ DISCIPLINE — think in years, not days ════════
- React to thesis changes, not short-term price wiggles or headlines-of-the-day.
- Add to winners, cut structural losers, keep some dry powder for opportunities.
- Explain WHY for every move. If the portfolio is already near target, do nothing.

════════ RUNTIME GUARDRAILS — enforced automatically ════════
- No single symbol may exceed 25% of equity; gross exposure capped by the leverage limit.
- Sells trim or close existing positions.

If the portfolio is already well-positioned, output "no changes" and call no
tools. This is a paper account: invest to maximize long-term simulated return.
"""


@dataclass
class _RebProposal:
    symbol: str
    side: str  # 'buy' | 'sell'
    target_weight_pct: float
    thesis: str = ""


class LongTermAgent(AgentBase):
    name = "long"
    sub_account = "long"

    def __init__(self, conn: sqlite3.Connection, broker: BrokerBase,
                 long_term: LongTermClient, provider: LLMProvider | None = None,
                 *, now: datetime | None = None) -> None:
        super().__init__(conn, broker, provider)
        self.long_term = long_term
        self._now = now

    def _wall(self) -> datetime:
        return self._now or now_utc()

    def run_once(self) -> RunOutcome:
        start = time_ms()
        if self._kill_switched():
            rid = self._record_run(status="halted", prompt="", response=None,
                                    tools_called=None, decisions=None,
                                    error="kill_switch", latency_ms=time_ms() - start)
            return RunOutcome(status="halted", error="kill_switch", run_id=rid)

        # Throttle: skip if rebalanced in the last 7 days AND no symbol drifted > 10%.
        last_run = self._last_rebalance_ts()
        wall = self._wall()
        within_interval = (last_run is not None
                            and (wall - last_run) < timedelta(days=REBALANCE_INTERVAL_DAYS))
        if within_interval and not self._has_significant_drift():
            rid = self._record_run(status="no-op", prompt="throttled", response=None,
                                    tools_called=None, decisions=None, error=None,
                                    latency_ms=time_ms() - start)
            return RunOutcome(status="no-op", run_id=rid)

        proposals, loop_res = self._run_llm_loop()
        decisions = self._size_and_validate(proposals)
        orders = self._place(decisions)

        rid = self._record_run(
            status="ok", prompt=SYSTEM_PROMPT[:200], response=loop_res.final_text,
            tools_called=[asdict_step(s) for s in loop_res.steps],
            decisions=[d.__dict__ for d in decisions],
            error=None, latency_ms=time_ms() - start,
        )
        return RunOutcome(status="ok", decisions=[d.__dict__ for d in decisions],
                          orders=orders, run_id=rid)

    # ----- throttle / drift -----

    def _last_rebalance_ts(self) -> datetime | None:
        aid = self._account_id()
        row = self.conn.execute(
            "SELECT ts FROM agent_runs WHERE account_id=? AND agent='long' AND status='ok' "
            "ORDER BY ts DESC LIMIT 1",
            (aid,),
        ).fetchone()
        if not row:
            return None
        try:
            return datetime.fromisoformat(row["ts"])
        except ValueError:
            return None

    def _has_significant_drift(self) -> bool:
        positions = self.broker.list_positions(self.sub_account)
        if not positions:
            return True  # never invested → always considered drifted
        acct = self.broker.get_account(self.sub_account)
        if acct.equity <= 0:
            return False
        for p in positions:
            weight = 100.0 * p.qty * p.mark_price / acct.equity
            if weight > MAX_SYMBOL_PCT + DRIFT_OVERRIDE_PCT:
                return True
        return False

    # ----- LLM tools -----

    def _run_llm_loop(self):
        proposals: list[_RebProposal] = []

        def get_recommendations(*, top_n: int = 18) -> dict:
            try:
                return self.long_term.get_recommendations(universe="Curated", top_n=top_n)
            except Exception as e:
                return {"error": str(e)}

        def lookup_ticker(symbol: str) -> dict:
            try:
                return self.long_term.lookup_ticker(symbol)
            except Exception as e:
                return {"error": str(e)}

        def get_news(symbol: str, *, days: int = 14) -> dict:
            try:
                return self.long_term.get_news(symbol, days=days, limit=20)
            except Exception as e:
                return {"error": str(e)}

        def current_positions() -> list[dict]:
            return [p.__dict__ for p in self.broker.list_positions(self.sub_account)]

        def account_snapshot() -> dict:
            return self.broker.get_account(self.sub_account).__dict__

        def propose_rebalance(*, symbol: str, side: str, target_weight_pct: float,
                              thesis: str = "") -> dict:
            proposals.append(_RebProposal(symbol=symbol.upper(), side=side,
                                            target_weight_pct=float(target_weight_pct),
                                            thesis=thesis))
            return {"ok": True, "buffered": len(proposals)}

        handlers = [
            ToolHandler(spec=ToolSpec(
                name="get_recommendations",
                description="Get the upstream long-horizon recommendations.",
                json_schema={"type": "object", "properties": {
                    "top_n": {"type": "integer", "default": 18}}}),
                fn=get_recommendations),
            ToolHandler(spec=ToolSpec(
                name="lookup_ticker", description="Fundamental/quote lookup for a symbol.",
                json_schema={"type": "object", "required": ["symbol"],
                              "properties": {"symbol": {"type": "string"}}}),
                fn=lookup_ticker),
            ToolHandler(spec=ToolSpec(
                name="get_news", description="Recent news for a symbol.",
                json_schema={"type": "object", "required": ["symbol"], "properties": {
                    "symbol": {"type": "string"},
                    "days": {"type": "integer", "default": 14}}}),
                fn=get_news),
            ToolHandler(spec=ToolSpec(
                name="current_positions", description="List currently held long-term positions.",
                json_schema={"type": "object"}), fn=current_positions),
            ToolHandler(spec=ToolSpec(
                name="account_snapshot", description="Equity, cash, positions value.",
                json_schema={"type": "object"}), fn=account_snapshot),
            ToolHandler(spec=ToolSpec(
                name="propose_rebalance",
                description=("Buffer a rebalance leg. side='buy' means lift weight to "
                              "target_weight_pct of equity; side='sell' trims/closes."),
                json_schema={"type": "object",
                              "required": ["symbol", "side", "target_weight_pct"],
                              "properties": {
                                  "symbol": {"type": "string"},
                                  "side": {"type": "string", "enum": ["buy", "sell"]},
                                  "target_weight_pct": {"type": "number"},
                                  "thesis": {"type": "string"}}}),
                fn=propose_rebalance),
        ]

        user = (
            f"It is {self._wall().isoformat()}. Review the long-term sub-account.\n"
            "Workflow: inspect `current_positions` and `account_snapshot`, pull the "
            "latest `get_recommendations`, and for any name you are considering, "
            "check `lookup_ticker` (fundamentals) and `get_news` (developments + "
            "sentiment). Then propose only the rebalance legs that materially improve "
            "the portfolio toward its targets. If it is already well-positioned, do nothing.")
        loop_res = self._run_llm(SYSTEM_PROMPT, user, handlers, max_steps=8)
        return proposals, loop_res

    def _size_and_validate(self, proposals: list[_RebProposal]) -> list[Decision]:
        acct = self.broker.get_account(self.sub_account)
        positions = {p.symbol: p for p in self.broker.list_positions(self.sub_account)}
        out: list[Decision] = []

        for p in proposals:
            d = Decision(symbol=p.symbol, side=p.side, qty=0.0, thesis=p.thesis)
            if acct.equity <= 0:
                d.accepted = False
                d.reject_reason = "zero_equity"
                out.append(d)
                continue
            # Clamp target to per-symbol cap.
            tgt_pct = max(0.0, min(MAX_SYMBOL_PCT, p.target_weight_pct))
            target_value = acct.equity * tgt_pct / 100.0
            held = positions.get(p.symbol)
            held_value = held.qty * held.mark_price if held else 0.0
            ref_price = held.mark_price if held else None

            if ref_price is None or ref_price <= 0:
                # No mark — fetch one from the upstream lookup.
                try:
                    info = self.long_term.lookup_ticker(p.symbol)
                    ref_price = float(info.get("price") or info.get("last") or 0.0)
                except Exception:
                    ref_price = 0.0
            if not ref_price or ref_price <= 0:
                d.accepted = False
                d.reject_reason = "no_price"
                out.append(d)
                continue

            if p.side == "buy":
                delta_value = max(0.0, target_value - held_value)
                qty = math.floor(delta_value / ref_price)
            else:  # sell
                delta_value = max(0.0, held_value - target_value)
                qty = math.floor(delta_value / ref_price)
                # If target_weight_pct == 0 close the whole position.
                if tgt_pct == 0 and held is not None:
                    qty = math.floor(held.qty)
            if qty <= 0:
                d.accepted = False
                d.reject_reason = "size_rounded_to_zero"
                out.append(d)
                continue
            d.qty = float(qty)
            d = validate(d, self.broker, self.sub_account, max_symbol_pct=MAX_SYMBOL_PCT)
            out.append(d)
        return out

    def _place(self, decisions: list[Decision]) -> list[dict]:
        orders: list[dict] = []
        for d in decisions:
            if not d.accepted:
                continue
            res = self.broker.place_order(OrderRequest(
                symbol=d.symbol, side=d.side, qty=d.qty, sub_account=self.sub_account,
                agent=self.name, thesis=d.thesis,
            ))
            orders.append({"id": res.id, "symbol": d.symbol, "status": res.status,
                            "fill_price": res.fill_price})
        return orders
