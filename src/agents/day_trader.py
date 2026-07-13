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
MAX_POSITION_PCT = 0.25           # cap any single position at 25% of equity
MAX_ORDER_USD = 25_000.0          # hard per-order notional cap (fits venue caps)

# Leveraged / inverse / vol ETFs that the upstream Alpaca MCP hard-blocks. The
# day agent avoids these for *stock* orders so trades actually land on Alpaca;
# leveraged directional exposure can still be taken via options (calls/puts).
# Union of the Alpaca MCP blocklist and the short-term scanner's leveraged
# universe so nothing the scanner can surface slips through.
ALPACA_BLOCKED_EQUITIES = frozenset({
    "TQQQ", "SQQQ", "SOXL", "SOXS", "FAS", "FAZ", "TNA", "TZA",
    "UPRO", "SPXU", "SPXL", "UDOW", "SDOW", "LABU", "LABD", "NUGT", "DUST",
    "JNUG", "JDST", "ERX", "ERY", "YINN", "YANG", "BOIL", "KOLD",
    "GUSH", "DRIP", "URTY", "SRTY", "TMF", "TMV", "TECL", "TECS",
    "UCO", "SCO", "QLD", "QID", "SSO", "SDS",
    "UVXY", "VXX", "SVXY", "VIXY",
    "TSLL", "TSLQ", "TSLT", "TSLZ", "NVDL", "NVDS", "NVDU", "AAPU", "AAPD",
    "MSFU", "MSFD", "AMZU", "AMZD", "METU", "BRKU", "CONL", "MSTU", "MSTX",
    "SH", "DOG", "DXD", "RWM", "PSQ",
})

SYSTEM_PROMPT = """You are "Atlas-Day", an elite intraday trader running a paper trading account.
Trade like a seasoned, self-aware human day-trader whose single objective is to
GROW the account's equity today while strictly protecting capital. You are invoked
about once per minute during market hours; doing nothing is a valid, common, and
often correct outcome — act only when you have a genuine, evidence-backed edge.

════════ DECISION FRAMEWORK — weigh ALL of these before acting ════════
1. TECHNICAL SETUP (primary trigger). Look at trend, momentum, support/resistance,
   volume, and the ATR-based stop the idea carries. A tradable setup needs BOTH a
   clear entry trigger AND a defined invalidation level (the stop).
2. NEWS & CATALYSTS. Call `get_news` on every serious candidate. Material, fresh
   news (earnings, guidance, upgrades/downgrades, M&A, legal, product, macro) can
   create or destroy an edge. Never trade INTO strong opposing news; prefer trades
   where price action and the news point the same way.
3. SENTIMENT. Use the aggregated multi-source sentiment from `get_news` as a
   tie-breaker and a read on crowd positioning — ride confirmation, fade extremes.
4. RISK / REWARD. Only take entries whose expected reward is at least ~2x the risk
   to your stop. Pass on marginal, coin-flip trades.
5. PORTFOLIO CONTEXT. Check `current_positions` and `account_snapshot`. Avoid
   over-concentration, duplicated exposure, and remember you can hold at most 5
   positions at once.

════════ HOW TO ACT ════════
- LONG  (side='buy')  — profit if price rises; stop is BELOW entry.
- SHORT (side='sell') — profit if price falls; stop is ABOVE entry.
- Leveraged / inverse ETFs (TQQQ, SQQQ, SOXL, …) are allowed for sharper
  directional bets when the setup and catalyst justify the extra volatility.
- OPTIONS — for defined-risk or higher-conviction directional plays, call
  `list_option_contracts` then `propose_option` (buy a CALL = bullish, a PUT =
  bearish). Keep options to 1-2 contracts given their leverage.
- Call `propose_trade` / `propose_option` once per intended entry. Every thesis
  MUST state: (a) the setup / trigger, (b) why the stop sits where it does,
  (c) the news + sentiment read, and (d) the rough risk:reward.

════════ TRADER DISCIPLINE — think like a pro, avoid rookie mistakes ════════
- Cut losers fast — that is what the stop is for. Let winners run.
- Do NOT chase extended moves or trade out of FOMO or boredom.
- Quality over quantity: 0-3 genuine setups per tick. Never force a trade.
- Avoid illiquid names and directionless chop where you have no edge.
- One thesis, one position — do not pyramid into a losing idea.

════════ RUNTIME GUARDRAILS — enforced automatically; you needn't compute them ════════
- Positions are sized for ~1% account risk via your stop and capped by per-order
  and per-symbol notional limits.
- Max 5 concurrent positions; gross exposure capped by the leverage limit.
- All positions are force-closed by 15:55 ET — do NOT open new trades after 15:30 ET.

If nothing offers a clear edge this minute, output "no trades today" and call no
tools. This is a paper account: trade to learn and to maximize simulated return.
"""


@dataclass
class _Proposal:
    symbol: str
    entry_price: float
    stop_price: float
    side: str = "buy"  # 'buy' (long) | 'sell' (short)
    thesis: str = ""


@dataclass
class _OptionProposal:
    occ_symbol: str
    qty: int
    side: str = "buy"  # 'buy' (open long call/put) | 'sell'
    thesis: str = ""


class DayTraderAgent(AgentBase):
    name = "day"
    sub_account = "day"

    def __init__(self, conn: sqlite3.Connection, broker: BrokerBase,
                 short_term: ShortTermClient, provider: LLMProvider | None = None,
                 *, now: datetime | None = None, options=None) -> None:
        super().__init__(conn, broker, provider)
        self.short_term = short_term
        self._now = now  # injectable for tests
        self.options = options  # AlpacaOptions | None — enables call/put trading

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
        proposals, option_proposals, loop_res = self._run_llm_loop()
        decisions = self._validate_and_size(proposals)
        orders = self._place(decisions)
        option_orders = self._place_options(option_proposals)

        all_decisions = [d.__dict__ for d in decisions] + option_orders
        rid = self._record_run(
            status="ok", prompt=SYSTEM_PROMPT[:200], response=loop_res.final_text,
            tools_called=[asdict_step(s) for s in loop_res.steps],
            decisions=all_decisions,
            error=None, latency_ms=time_ms() - start,
        )
        return RunOutcome(status="ok", decisions=all_decisions,
                          orders=orders + option_orders, run_id=rid)

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
        option_proposals: list[_OptionProposal] = []

        def list_intraday_ideas(*, tier: str = "A", limit: int = 10) -> dict:
            try:
                res = self.short_term.list_ideas(mode="intraday", tier=tier, limit=limit)
            except Exception as e:
                return {"error": str(e)}
            # Drop symbols the Alpaca leg will refuse (leveraged/inverse/vol ETFs)
            # so proposed stock trades can actually execute on Alpaca.
            key = "rows" if "rows" in res else ("ideas" if "ideas" in res else None)
            if key:
                kept = [r for r in (res.get(key) or [])
                        if str(r.get("ticker") or r.get("symbol") or "").upper()
                        not in ALPACA_BLOCKED_EQUITIES]
                res = {**res, key: kept, "count": len(kept)}
            return res

        def get_quote(symbol: str) -> dict:
            try:
                return self.short_term.lookup_ticker(symbol, interval="5m", period="1d")
            except Exception as e:
                return {"error": str(e)}

        def get_news(symbol: str, *, days: int = 2) -> dict:
            try:
                return self.short_term.get_news(symbol, days=days, limit=15)
            except Exception as e:
                return {"error": str(e)}

        def current_positions() -> list[dict]:
            return [p.__dict__ for p in self.broker.list_positions(self.sub_account)]

        def account_snapshot() -> dict:
            return self.broker.get_account(self.sub_account).__dict__

        def propose_trade(*, symbol: str, entry_price: float, stop_price: float,
                          side: str = "buy", thesis: str = "") -> dict:
            proposals.append(_Proposal(symbol=symbol.upper(),
                                         entry_price=float(entry_price),
                                         stop_price=float(stop_price),
                                         side=("sell" if str(side).lower() in ("sell", "short")
                                               else "buy"),
                                         thesis=thesis))
            return {"ok": True, "buffered": len(proposals)}

        def list_option_contracts(*, underlying: str, option_type: str = "call",
                                   limit: int = 15) -> dict:
            if self.options is None:
                return {"error": "options_unavailable"}
            try:
                cs = self.options.find_contracts(underlying, option_type=option_type,
                                                  limit=limit)
                return {"count": len(cs), "contracts": cs}
            except Exception as e:
                return {"error": str(e)}

        def propose_option(*, occ_symbol: str, qty: int = 1, side: str = "buy",
                           thesis: str = "") -> dict:
            option_proposals.append(_OptionProposal(
                occ_symbol=occ_symbol.upper(), qty=max(1, int(qty)),
                side=("sell" if str(side).lower() == "sell" else "buy"), thesis=thesis))
            return {"ok": True, "buffered": len(option_proposals)}

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
                name="get_news",
                description=("Recent news headlines + VADER sentiment for a symbol, "
                              "aggregated across Finnhub, Alpha Vantage, Marketaux, NewsAPI, "
                              "Tiingo, Yahoo, StockTwits, SEC filings and Reddit. Use it to "
                              "confirm or veto a technical setup with the news/sentiment "
                              "backdrop."),
                json_schema={"type": "object", "required": ["symbol"], "properties": {
                    "symbol": {"type": "string"},
                    "days": {"type": "integer", "default": 2}}}), fn=get_news),
            ToolHandler(spec=ToolSpec(
                name="current_positions", description="List currently held day-trading positions.",
                json_schema={"type": "object"}), fn=current_positions),
            ToolHandler(spec=ToolSpec(
                name="account_snapshot", description="Equity, cash, positions value.",
                json_schema={"type": "object"}), fn=account_snapshot),
            ToolHandler(spec=ToolSpec(
                name="propose_trade",
                description=("Buffer an entry proposal. side='buy' opens a long (stop below "
                              "entry); side='sell' opens a short (stop above entry). Runtime "
                              "sizes it for 1% account risk based on |entry - stop| and "
                              "places the order."),
                json_schema={"type": "object",
                              "required": ["symbol", "entry_price", "stop_price"],
                              "properties": {
                                  "symbol": {"type": "string"},
                                  "side": {"type": "string", "enum": ["buy", "sell"],
                                            "default": "buy"},
                                  "entry_price": {"type": "number"},
                                  "stop_price": {"type": "number"},
                                  "thesis": {"type": "string"}}}),
                fn=propose_trade),
        ]

        if self.options is not None:
            handlers += [
                ToolHandler(spec=ToolSpec(
                    name="list_option_contracts",
                    description=("List tradable option contracts (calls or puts) for an "
                                  "underlying on Alpaca paper. Use to pick an OCC symbol for "
                                  "`propose_option`."),
                    json_schema={"type": "object", "required": ["underlying"], "properties": {
                        "underlying": {"type": "string"},
                        "option_type": {"type": "string", "enum": ["call", "put"],
                                         "default": "call"},
                        "limit": {"type": "integer", "default": 15}}}),
                    fn=list_option_contracts),
                ToolHandler(spec=ToolSpec(
                    name="propose_option",
                    description=("Buffer an options order. occ_symbol is an OCC option symbol "
                                  "from `list_option_contracts` (e.g. 'AAPL250620C00190000'). "
                                  "side='buy' opens a long call/put; qty is number of contracts "
                                  "(1 = 100 shares). Placed directly on Alpaca paper."),
                    json_schema={"type": "object", "required": ["occ_symbol"], "properties": {
                        "occ_symbol": {"type": "string"},
                        "qty": {"type": "integer", "default": 1},
                        "side": {"type": "string", "enum": ["buy", "sell"], "default": "buy"},
                        "thesis": {"type": "string"}}}),
                    fn=propose_option),
            ]

        opt_hint = ("" if self.options is None else
                    " Options (calls/puts) are also available via `list_option_contracts` "
                    "then `propose_option` for defined-risk or higher-conviction plays.")
        user = (
            f"It is {self._wall().isoformat()} (US market hours). Manage the day-trading "
            "sub-account for maximum return today.\n"
            "Workflow this tick:\n"
            "1. Call `list_intraday_ideas` to see today's candidates.\n"
            "2. For the most promising 1-3, confirm with `get_quote` (price/vol) and "
            "`get_news` (catalysts + sentiment).\n"
            "3. Check `current_positions` / `account_snapshot` for context and room.\n"
            "4. Propose only setups with a clear trigger, a defined stop, and >=2:1 "
            "reward:risk — long, short, leveraged ETF, or option. Otherwise do nothing."
            + opt_hint)
        loop_res = self._run_llm(SYSTEM_PROMPT, user, handlers, max_steps=8)
        return proposals, option_proposals, loop_res

    def _place_options(self, proposals: list[_OptionProposal]) -> list[dict]:
        """Place buffered option orders directly on Alpaca; record + return each."""
        from src.brokers.alpaca_options import OptionsRecorder
        out: list[dict] = []
        if self.options is None or not proposals:
            return out
        recorder = OptionsRecorder(self.conn)
        for p in proposals:
            try:
                resp = self.options.place_order(occ_symbol=p.occ_symbol, qty=p.qty, side=p.side)
                oid = recorder.record(sub_account=self.sub_account, occ_symbol=p.occ_symbol,
                                       side=p.side, qty=p.qty, agent=self.name,
                                       thesis=p.thesis, resp=resp)
                out.append({"id": oid, "symbol": p.occ_symbol, "side": p.side, "qty": p.qty,
                            "instrument": "option", "status": str(resp.get("status", "")),
                            "external_id": resp.get("id"), "thesis": p.thesis, "accepted": True})
            except Exception as e:  # noqa: BLE001
                log.exception("option order failed for %s", p.occ_symbol)
                oid = recorder.record(sub_account=self.sub_account, occ_symbol=p.occ_symbol,
                                       side=p.side, qty=p.qty, agent=self.name,
                                       thesis=p.thesis, resp=None, error=str(e)[:200])
                out.append({"id": oid, "symbol": p.occ_symbol, "side": p.side, "qty": p.qty,
                            "instrument": "option", "status": "rejected",
                            "thesis": p.thesis, "accepted": False, "reject_reason": str(e)[:120]})
        return out

    def _validate_and_size(self, proposals: list[_Proposal]) -> list[Decision]:
        acct = self.broker.get_account(self.sub_account)
        open_syms = {p.symbol for p in self.broker.list_positions(self.sub_account) if p.qty > 0}
        out: list[Decision] = []
        slots_left = MAX_CONCURRENT_POSITIONS - len(open_syms)

        for p in proposals:
            d = Decision(symbol=p.symbol, side=p.side, qty=0.0, thesis=p.thesis)
            is_short = p.side == "sell"
            if p.symbol in ALPACA_BLOCKED_EQUITIES:
                d.accepted = False
                d.reject_reason = "alpaca_blocked_etf"
                out.append(d)
                continue
            if slots_left <= 0 and p.symbol not in open_syms:
                d.accepted = False
                d.reject_reason = "max_concurrent_positions"
                out.append(d)
                continue
            # Risk per share = distance to stop. Long: entry>stop. Short: stop>entry.
            risk_per_share = abs(p.entry_price - p.stop_price)
            if risk_per_share < 1e-6:
                d.accepted = False
                d.reject_reason = "stop_equals_entry"
                out.append(d)
                continue
            # For a long the stop must be below entry; for a short, above.
            if (not is_short and p.stop_price >= p.entry_price) or \
               (is_short and p.stop_price <= p.entry_price):
                d.accepted = False
                d.reject_reason = "stop_on_wrong_side"
                out.append(d)
                continue
            risk_budget = acct.equity * ACCOUNT_RISK_PCT
            qty = math.floor(risk_budget / risk_per_share)
            # Cap by notional so tight stops on cheap/vol names don't produce
            # absurd share counts that blow past per-order / per-symbol caps.
            # Also respect the Alpaca MCP per-order cap so the primary leg fills.
            if p.entry_price > 0:
                from src.config import get_settings
                try:
                    alpaca_cap = float(get_settings().stock_rec_max_order_usd or MAX_ORDER_USD)
                except (TypeError, ValueError):
                    alpaca_cap = MAX_ORDER_USD
                order_cap = min(MAX_ORDER_USD, alpaca_cap)
                max_by_pct = math.floor((acct.equity * MAX_POSITION_PCT) / p.entry_price)
                max_by_order = math.floor(order_cap / p.entry_price)
                qty = min(qty, max_by_pct, max_by_order)
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
