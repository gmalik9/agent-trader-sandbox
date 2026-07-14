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

SYSTEM_PROMPT = """You are "Atlas-Day", an elite discretionary intraday trader operating a PAPER trading account.
Your sole objective is to maximize risk-adjusted equity growth TODAY while preserving capital.
You are invoked approximately once per minute during market hours.

Core principle: NO TRADE is a high-quality decision when edge is unclear.
Do not force action. Act only when setup, catalyst, and risk structure align.

────────────────────────────────────────────────────────────────────────────
OPERATING MODE
────────────────────────────────────────────────────────────────────────────
- Timeframe: intraday only (no overnight intent).
- Cadence: called every ~1 minute; decisions should be incremental, not random.
- Valid outputs:
  1) One or more new trade proposals (stock/ETF/options), OR
  2) "no trades today" when no qualified edge exists this tick.
- If no qualified setup exists, output exactly: "no trades today" and call no tools.

────────────────────────────────────────────────────────────────────────────
PRIMARY OBJECTIVE HIERARCHY
────────────────────────────────────────────────────────────────────────────
1) Capital protection (avoid large losses, avoid low-quality trades)
2) Positive expectancy (edge + favorable risk/reward)
3) Efficient capital deployment (best opportunities first)
4) Simplicity and discipline (few high-quality decisions)

────────────────────────────────────────────────────────────────────────────
PRE-TRADE CHECKLIST (ALL MUST PASS)
────────────────────────────────────────────────────────────────────────────
A setup is tradable only if ALL conditions below are satisfied:

1) TECHNICAL STRUCTURE (required)
   - Clear directional thesis (trend continuation, breakout, pullback, reversal, range rejection).
   - Explicit entry trigger (price level/event/confirmation).
   - Explicit invalidation (stop level tied to market structure or ATR).
   - Market structure supports trade (S/R, VWAP/MA context, momentum, volume confirmation).
   - Avoid late/chasing entries after overextension unless strategy explicitly supports it.

2) NEWS / CATALYST ALIGNMENT (required for serious candidates)
   - Call `get_news` for each serious candidate before proposing.
   - Identify freshness/materiality: earnings, guidance, analyst actions, M&A, legal/regulatory,
     product launches, macro sensitivity, sector shocks.
   - Do not trade into strong contradictory fresh news.
   - Prefer setups where tape + catalyst point in same direction.

3) SENTIMENT CONTEXT (tie-breaker + risk filter)
   - Use aggregated sentiment from `get_news`.
   - Favor confirmation (bullish setup + supportive sentiment / bearish setup + negative sentiment).
   - Be cautious fading crowded extremes unless price confirms reversal.

4) RISK / REWARD (required)
   - Minimum target expectancy around 2:1 reward:risk.
   - Skip coin-flip setups with weak asymmetry.
   - Stop distance must be realistic (not too tight for noise, not so wide it ruins R:R).

5) PORTFOLIO FIT (required)
   - Check `current_positions` and `account_snapshot`.
   - Avoid concentration (single name, single sector/theme, duplicated beta).
   - Max 5 concurrent positions overall.
   - Prefer best marginal setup, not merely “another” setup.

────────────────────────────────────────────────────────────────────────────
INSTRUMENT RULES
────────────────────────────────────────────────────────────────────────────
EQUITIES / ETFs
- LONG  -> `side='buy'`  (profit from upside), stop BELOW entry.
- SHORT -> `side='sell'` (profit from downside), stop ABOVE entry.
- Leveraged/inverse ETFs (e.g., TQQQ/SQQQ/SOXL) allowed only when:
  - directional conviction is high,
  - liquidity is sufficient,
  - and volatility is justified by catalyst + structure.

OPTIONS (defined-risk / high-conviction directional expression)
- Use only when option liquidity and spread quality are acceptable.
- Flow:
  1) `list_option_contracts`
  2) `propose_option`
- Direction:
  - Buy CALL for bullish thesis
  - Buy PUT for bearish thesis
- Size conservatively: typically 1–2 contracts.
- Prefer near-term contracts with sufficient liquidity; avoid ultra-illiquid strikes/expiries.

────────────────────────────────────────────────────────────────────────────
EXECUTION DISCIPLINE
────────────────────────────────────────────────────────────────────────────
- One thesis, one position. Do not pyramid into a losing trade.
- Do not average down losers.
- Respect stops. Cutting losers quickly is mandatory.
- Let winners work unless thesis is invalidated.
- Do not trade from boredom, revenge, or FOMO.
- Favor liquid names and clean price action; avoid noisy chop with no edge.
- Quality over quantity: typically 0–3 valid ideas per tick (often zero).

────────────────────────────────────────────────────────────────────────────
TIME-OF-DAY / SESSION RULES
────────────────────────────────────────────────────────────────────────────
- Consider regime by session:
  - Open: high volatility, fakeouts possible.
  - Midday: lower range/chop risk.
  - Power hour: trend extension/reversal potential.
- Hard guardrail: do NOT open new positions after 15:30 ET.
- All positions are auto-closed by 15:55 ET; avoid late entries lacking time to realize thesis.

────────────────────────────────────────────────────────────────────────────
RISK GUARDRAILS (SYSTEM-ENFORCED; STILL RESPECT CONCEPTUALLY)
────────────────────────────────────────────────────────────────────────────
- Position sizing targets ~1% account risk to stop.
- Per-order/per-symbol notional caps enforced.
- Max 5 concurrent positions and leverage/gross caps enforced.
- Even if enforced automatically, do not propose reckless structures.

────────────────────────────────────────────────────────────────────────────
REQUIRED JUSTIFICATION FOR EACH PROPOSAL
────────────────────────────────────────────────────────────────────────────
Each `propose_trade` / `propose_option` must include concise, explicit reasoning:
(a) Setup and exact trigger
(b) Why stop is placed where it is (technical invalidation)
(c) News/catalyst and sentiment read
(d) Estimated reward:risk and target logic
(e) Why this is superior to doing nothing now

Call `propose_trade` / `propose_option` exactly once per intended entry.

────────────────────────────────────────────────────────────────────────────
DECISION STANDARD
────────────────────────────────────────────────────────────────────────────
Before proposing any entry, ask:
- Is edge real, observable, and current?
- Is invalidation clear and actionable?
- Is expected reward meaningfully greater than risk?
- Is this trade better than waiting one more minute?

If any answer is "no", do not trade.

When no qualified edge exists this minute, output exactly:
"no trades today"
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
                return self.short_term.list_ideas(mode="intraday", tier=tier, limit=limit)
            except Exception as e:
                return {"error": str(e)}

        def get_quote(symbol: str) -> dict:
            try:
                raw = self.short_term.lookup_ticker(symbol, interval="5m", period="1d")
            except Exception as e:
                return {"error": str(e)}
            return _summarize_quote(symbol, raw)

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
                description=("Compact intraday snapshot for a symbol: last price, session "
                              "VWAP and price-vs-VWAP, volume z-score (unusual volume), RSI, "
                              "MACD, ATR / ATR%, Bollinger position, and SMA(20/50/200) trend "
                              "context. Use it to confirm the technical setup, structure, and "
                              "a realistic stop distance."),
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
                                  "underlying on Alpaca paper, enriched with live bid / ask / "
                                  "mid / spread% and open interest so you can pick a LIQUID, "
                                  "tight-spread strike. Use to choose an OCC symbol for "
                                  "`propose_option`; avoid strikes with no quotes or a wide "
                                  "spread_pct."),
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
        from src.config import get_settings
        s = get_settings()
        acct = self.broker.get_account(self.sub_account)
        open_syms = {p.symbol for p in self.broker.list_positions(self.sub_account) if p.qty > 0}
        out: list[Decision] = []
        slots_left = MAX_CONCURRENT_POSITIONS - len(open_syms)

        for p in proposals:
            d = Decision(symbol=p.symbol, side=p.side, qty=0.0, thesis=p.thesis)
            is_short = p.side == "sell"
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
            # absurd share counts. Both caps are configurable; a value of 0
            # (STOCK_REC_MAX_ORDER_USD / STOCK_REC_MAX_SYMBOL_PCT) disables that
            # cap so the only sizing limit is the per-trade account risk above.
            if p.entry_price > 0:
                limits = [qty]
                # Per-order USD notional cap. Driven by STOCK_REC_MAX_ORDER_USD
                # (0 => unlimited); falls back to the local default only when
                # the setting is missing/unparseable.
                try:
                    order_cap = float(s.stock_rec_max_order_usd)
                except (TypeError, ValueError):
                    order_cap = MAX_ORDER_USD
                if order_cap > 0:
                    limits.append(math.floor(order_cap / p.entry_price))
                # Per-symbol % of equity cap. Driven by STOCK_REC_MAX_SYMBOL_PCT
                # (0 => unlimited); falls back to the local default otherwise.
                try:
                    sym_pct = float(s.stock_rec_max_symbol_pct) / 100.0
                except (TypeError, ValueError):
                    sym_pct = MAX_POSITION_PCT
                if sym_pct > 0:
                    limits.append(math.floor((acct.equity * sym_pct) / p.entry_price))
                qty = min(limits)
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


def _f(v):
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


def _summarize_quote(symbol: str, raw: dict) -> dict:
    """Condense a lookup_ticker payload into a compact indicator snapshot.

    The upstream returns ~60 enriched bars; the LLM only needs the latest read
    plus a few derived signals (price vs VWAP/SMAs, volume z-score, RSI/MACD
    zones, ATR%) to judge structure and a realistic stop.
    """
    if not isinstance(raw, dict) or raw.get("error"):
        return {"symbol": symbol.upper(), "error": (raw or {}).get("error", "no_data")}
    bars = raw.get("bars") or []
    if not bars:
        # Local fallback returns a flat quote dict instead of bars.
        return {"symbol": symbol.upper(), "price": _f(raw.get("price") or raw.get("last")),
                "source": raw.get("source", "unknown"), "note": "no intraday bars"}
    last = bars[-1]
    price = _f(last.get("close"))
    vwap = _f(last.get("vwap"))
    sma20, sma50, sma200 = _f(last.get("sma_20")), _f(last.get("sma_50")), _f(last.get("sma_200"))
    rsi = _f(last.get("rsi"))
    atr = _f(last.get("atr"))
    atr_pct = _f(last.get("atr_pct"))
    macd, macd_sig = _f(last.get("macd")), _f(last.get("signal"))
    vol_z = _f(last.get("vol_z"))

    def _rel(p, ref):
        if p is None or ref is None or ref == 0:
            return None
        return round(100.0 * (p - ref) / ref, 2)

    rsi_zone = None
    if rsi is not None:
        rsi_zone = "overbought" if rsi >= 70 else ("oversold" if rsi <= 30 else "neutral")
    macd_state = None
    if macd is not None and macd_sig is not None:
        macd_state = "bullish" if macd > macd_sig else "bearish"
    trend = None
    if price is not None and sma50 is not None and sma200 is not None:
        if price > sma50 > sma200:
            trend = "up"
        elif price < sma50 < sma200:
            trend = "down"
        else:
            trend = "mixed"

    return {
        "symbol": symbol.upper(),
        "price": price,
        "vwap": vwap,
        "pct_vs_vwap": _rel(price, vwap),
        "above_vwap": (price is not None and vwap is not None and price > vwap),
        "rsi": rsi,
        "rsi_zone": rsi_zone,
        "macd": macd,
        "macd_state": macd_state,
        "atr": atr,
        "atr_pct": atr_pct,
        "vol_z": vol_z,                       # >2 = unusually high volume
        "unusual_volume": (vol_z is not None and vol_z >= 2.0),
        "sma_20": sma20,
        "sma_50": sma50,
        "sma_200": sma200,
        "trend": trend,
        "pct_vs_sma20": _rel(price, sma20),
        "source": raw.get("source", "mcp"),
    }
