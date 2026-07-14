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
from src.llm.github_models import RateLimitError
from src.llm.tool_loop import ToolHandler
from src.mcp_clients.short_term import ShortTermClient
from src.sandbox import db as dbm
from src.sandbox.clock import is_force_flat_window, is_market_open

log = logging.getLogger(__name__)

MAX_CONCURRENT_POSITIONS = 8      # allow diversification across uncorrelated names
ACCOUNT_RISK_PCT = 0.01           # 1% of equity per trade
DD_HALT_PCT = -2.0                # halt for the day at -2% intraday
DEFAULT_ATR_PCT = 0.02            # 2% fallback ATR when upstream doesn't return one
MAX_POSITION_PCT = 0.25           # cap any single position at 25% of equity
MAX_ORDER_USD = 25_000.0          # hard per-order notional cap (fits venue caps)
DUST_POSITION_PCT = 0.02          # positions below 2% of equity don't consume a slot

# Leveraged / inverse ETF pairs. Alpaca refuses to *short* most leveraged and
# inverse ETFs (422 "cannot be sold short"), so a bearish view on one of these
# is expressed by BUYING (going long) its inverse counterpart instead. The map
# is bidirectional: shorting either leg is converted to a long of the other.
INVERSE_ETF: dict[str, str] = {
    # Nasdaq-100
    "TQQQ": "SQQQ", "SQQQ": "TQQQ", "QLD": "QID", "QID": "QLD",
    # S&P 500
    "UPRO": "SPXU", "SPXU": "UPRO", "SPXL": "SPXS", "SPXS": "SPXL",
    "SSO": "SDS", "SDS": "SSO",
    # Dow
    "UDOW": "SDOW", "SDOW": "UDOW",
    # Russell 2000
    "TNA": "TZA", "TZA": "TNA", "URTY": "SRTY", "SRTY": "URTY",
    # Semiconductors
    "SOXL": "SOXS", "SOXS": "SOXL",
    # Technology
    "TECL": "TECS", "TECS": "TECL",
    # Financials
    "FAS": "FAZ", "FAZ": "FAS",
    # Biotech
    "LABU": "LABD", "LABD": "LABU",
    # Gold miners
    "NUGT": "DUST", "DUST": "NUGT", "JNUG": "JDST", "JDST": "JNUG",
    # Energy
    "ERX": "ERY", "ERY": "ERX", "GUSH": "DRIP", "DRIP": "GUSH",
    # Oil / nat-gas
    "UCO": "SCO", "SCO": "UCO", "BOIL": "KOLD", "KOLD": "BOIL",
    # China
    "YINN": "YANG", "YANG": "YINN",
    # Treasuries
    "TMF": "TMV", "TMV": "TMF",
    # Volatility (long-vol vs short-vol)
    "UVXY": "SVXY", "VXX": "SVXY", "VIXY": "SVXY", "SVXY": "VXX",
    # Single-stock leveraged
    "TSLL": "TSLQ", "TSLQ": "TSLL", "NVDL": "NVDS", "NVDS": "NVDL",
}

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
   - Start from `list_intraday_ideas`: it is pre-ranked by a numeric conviction
     `heat_score` (0-100) that already blends momentum, volume, volatility AND
     news sentiment (the `news_spike` signal; `has_news_catalyst` flags it).
   - Prefer higher `heat_score`, tier A, and ideas WITH a news catalyst over a
     purely volatility-driven name (e.g. a leveraged ETF tagged only
     `atr_leader,gapper` with no news_spike is a low-quality, high-risk pick).
   - Clear directional thesis (trend continuation, breakout, pullback, reversal, range rejection).
   - Explicit entry trigger (price level/event/confirmation).
   - Explicit invalidation (stop level tied to market structure or ATR).
   - Market structure supports trade (S/R, VWAP/MA context, momentum, volume confirmation).
   - Avoid late/chasing entries after overextension unless strategy explicitly supports it.
   - Do NOT re-propose a name you already hold or that was already rejected this
     session; move down the ranked list to the next-best DIFFERENT setup.

2) NEWS / CATALYST ALIGNMENT (required for serious candidates)
   - Call `get_news` for each serious candidate before proposing.
   - `get_news` returns an aggregated `sentiment_score` (−1..+1), a
     `sentiment_label`, bullish/bearish article counts and the top headlines
     across 9 sources. Read it, don't just glance at it.
   - If the idea's `has_news_catalyst` is false, the ranking is volatility-driven,
     not news-driven — demand a stronger technical + sentiment confirmation.
   - Identify freshness/materiality: earnings, guidance, analyst actions, M&A, legal/regulatory,
     product launches, macro sensitivity, sector shocks.
   - Do not trade into strong contradictory fresh news.
   - Prefer setups where tape + catalyst point in same direction.

3) SENTIMENT & ANALYST CONTEXT (required directional cross-check)
   - Use the aggregated `sentiment_score` / `sentiment_label` from `get_news`.
   - Call `get_analyst_view` for each serious candidate: it returns the
     Wall-Street `rating` (Strong Buy … Strong Sell), consensus `target_price`,
     `analyst_count`, implied `upside_pct`, and aggregated sentiment.
   - Use these as a directional filter, NOT a standalone signal:
     • Do NOT short (or buy the inverse ETF of) a name with a Strong-Buy rating
       and large positive upside unless the tape strongly contradicts it.
     • Do NOT buy a name rated Sell/Strong-Sell with negative sentiment and
       downside to target unless there is a clear catalyst-driven reversal.
     • Best trades: technical setup, news sentiment AND analyst view all agree.
   - Favor confirmation; be cautious fading crowded extremes unless price confirms.

4) RISK / REWARD (required)
   - Minimum target expectancy around 2:1 reward:risk.
   - Skip coin-flip setups with weak asymmetry.
   - Stop distance must be realistic (not too tight for noise, not so wide it ruins R:R).

5) PORTFOLIO FIT (required)
   - Check `current_positions` and `account_snapshot`, and read the `portfolio`
     block returned by `list_intraday_ideas` (idle cash %, names already held,
     per-name room left).
   - DIVERSIFY and DEPLOY: if a large share of equity is idle cash, actively put
     it to work across SEVERAL uncorrelated high-conviction names — do not sit in
     cash and do not pile into one ticker. Aim to build a book of multiple
     positions, not a single concentrated bet.
   - Each idea shows `already_held_pct`, `room_left` and `at_cap`. NEVER re-propose
     a name with `at_cap=true` (it is already at its ~20% limit and will be
     rejected); move to the next-best DIFFERENT idea instead.
   - Avoid concentration (single name, single sector/theme, duplicated beta).
   - Single-name exposure is auto-capped near 20% of equity; you can hold up to
     8 concurrent positions — use that room to spread risk.
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
- IMPORTANT — expressing a bearish view on a leveraged/inverse ETF:
  the broker will NOT let you short leveraged or inverse ETFs. To be bearish
  on one, BUY (go long) its inverse counterpart instead. Examples:
    - bearish semis: buy SOXS (not short SOXL)
    - bearish Nasdaq: buy SQQQ (not short TQQQ)
    - bearish S&P: buy SPXU/SPXS   · bearish small-caps: buy TZA
    - bearish tech: buy TECS       · bearish financials: buy FAZ
  Set entry/stop on the ETF you are actually buying. If you do submit a short
  on a leveraged/inverse ETF, it is automatically converted to a long of its
  inverse with an equivalent percentage stop.

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
                 *, now: datetime | None = None, options=None, long_term=None) -> None:
        super().__init__(conn, broker, provider)
        self.short_term = short_term
        self.long_term = long_term  # LongTermClient | LocalLongTermClient | None
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
        try:
            proposals, option_proposals, loop_res = self._run_llm_loop()
        except RateLimitError as e:
            # Provider (and its fallback) are rate-limited this tick. Record a
            # visible no-op so idleness is explained rather than a silent crash /
            # gap in the run history.
            rid = self._record_run(status="no-op", prompt="rate_limited", response=None,
                                    tools_called=None, decisions=None,
                                    error=f"rate_limited:{str(e)[:160]}",
                                    latency_ms=time_ms() - start)
            return RunOutcome(status="no-op", error="rate_limited", run_id=rid)
        proposals = self._maybe_substitute_inverse(proposals)
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
            # Annotate with what we already hold and how much room is left per
            # name, and surface idle cash, so the LLM diversifies into NEW names
            # instead of re-proposing a symbol that's already at its cap.
            from src.config import get_settings
            s = get_settings()
            try:
                acct = self.broker.get_account(self.sub_account)
                positions = self.broker.list_positions(self.sub_account)
            except Exception:
                acct, positions = None, []
            equity = acct.equity if acct else 0.0
            cap_pct = float(getattr(s, "day_max_position_pct", MAX_POSITION_PCT) or 0.20)
            held = {}
            for p in positions:
                if p.qty > 0 and equity > 0:
                    held[p.symbol.upper()] = 100.0 * abs(p.qty) * (p.mark_price or 0.0) / equity
            ctx = {
                "equity": round(equity, 2) if equity else None,
                "cash": round(acct.cash, 2) if acct else None,
                "pct_cash_idle": round(100.0 * acct.cash / equity, 1) if (acct and equity) else None,
                "per_name_cap_pct": round(cap_pct * 100, 1),
                "held": {k: round(v, 1) for k, v in held.items()},
                "open_position_count": len(held),
                "max_positions": MAX_CONCURRENT_POSITIONS,
            }
            return _summarize_ideas(res, held_pct=held, cap_pct=cap_pct * 100, context=ctx)

        def get_quote(symbol: str) -> dict:
            try:
                raw = self.short_term.lookup_ticker(symbol, interval="5m", period="1d")
            except Exception as e:
                return {"error": str(e)}
            return _summarize_quote(symbol, raw)

        def get_news(symbol: str, *, days: int = 2) -> dict:
            return self._news(symbol, days=days)

        def get_analyst_view(symbol: str) -> dict:
            return self._analyst_view(symbol)

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
                description=("Ranked intraday trade ideas from the upstream scanner. Each "
                              "idea carries a numeric conviction `heat_score` (0-100) that "
                              "ALREADY folds in 9-source news sentiment (via the news_spike "
                              "signal), plus `tier` (A/B/C), `signal_tags` (which scanners "
                              "fired), `has_news_catalyst` (true if news moved the score), "
                              "and pre-computed entry/stop/target/rr/dollar_risk. Rank by "
                              "heat_score and PREFER ideas with a real news catalyst; "
                              "diversify across the best setups rather than repeating one "
                              "name."),
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
                description=("Recent news for a symbol with aggregated VADER sentiment, "
                              "across Finnhub, Alpha Vantage, Marketaux, NewsAPI, Tiingo, "
                              "Yahoo, StockTwits, SEC filings and Reddit. Returns a "
                              "sentiment_score (-1..1), sentiment_label, bullish/bearish "
                              "article counts and the top headlines. Use it to confirm or "
                              "veto a technical setup with the news/sentiment backdrop."),
                json_schema={"type": "object", "required": ["symbol"], "properties": {
                    "symbol": {"type": "string"},
                    "days": {"type": "integer", "default": 2}}}), fn=get_news),
            ToolHandler(spec=ToolSpec(
                name="get_analyst_view",
                description=("Analyst consensus and market-sentiment read for a symbol: "
                              "Wall-Street rating (Strong Buy … Strong Sell), consensus "
                              "price target, number of covering analysts, implied upside %, "
                              "plus aggregated news sentiment. Use it as a directional "
                              "cross-check — e.g. avoid shorting into a Strong-Buy with big "
                              "upside, or buying into a Sell with negative sentiment."),
                json_schema={"type": "object", "required": ["symbol"], "properties": {
                    "symbol": {"type": "string"}}}), fn=get_analyst_view),
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

    def _news(self, symbol: str, *, days: int = 2) -> dict:
        """News + aggregated sentiment for a symbol.

        Prefers the stock-recommender leg (9-source news WITH per-article VADER
        sentiment); falls back to the short-term leg (headlines only) when the
        long-term client is unavailable or returns nothing.
        """
        raw = None
        if self.long_term is not None:
            try:
                raw = self.long_term.get_news(symbol, days=days, limit=20)
            except Exception:
                log.debug("long_term.get_news failed for %s", symbol, exc_info=True)
                raw = None
        if not raw or raw.get("error") or not (raw.get("articles") or raw.get("news")):
            try:
                raw = self.short_term.get_news(symbol, days=days, limit=15)
            except Exception as e:
                return {"error": str(e)}
        return _summarize_news(symbol, raw)

    def _analyst_view(self, symbol: str) -> dict:
        """Analyst consensus + market-sentiment read from the stock-recommender leg."""
        if self.long_term is None:
            return {"error": "analyst_data_unavailable"}
        try:
            raw = self.long_term.lookup_ticker(symbol)
        except Exception as e:
            return {"error": str(e)}
        return _summarize_analyst(symbol, raw)

    def _maybe_substitute_inverse(self, proposals: list[_Proposal]) -> list[_Proposal]:
        """Convert shorts of leveraged/inverse ETFs into longs of their inverse.

        Alpaca refuses to short most leveraged/inverse ETFs. A short of such a
        symbol is rewritten as a BUY of its inverse counterpart, translating the
        entry/stop to the inverse instrument at an equivalent percentage stop so
        the 1%-risk sizing stays intact. Non-leveraged shorts pass through
        unchanged (Alpaca shorts ordinary shortable names fine).
        """
        out: list[_Proposal] = []
        for p in proposals:
            inv = INVERSE_ETF.get(p.symbol.upper())
            if p.side != "sell" or not inv or p.entry_price <= 0:
                out.append(p)
                continue
            # Percentage stop distance on the original (short: stop above entry).
            stop_pct = abs(p.stop_price - p.entry_price) / p.entry_price
            inv_price = self._latest_price(inv)
            if inv_price is None or inv_price <= 0:
                # Can't price the inverse; keep the original so it's visibly
                # rejected downstream rather than silently dropped.
                log.info("inverse substitution: no price for %s (short %s); leaving as-is",
                         inv, p.symbol)
                out.append(p)
                continue
            new_stop = round(inv_price * (1.0 - stop_pct), 2)  # long stop below entry
            if new_stop >= inv_price:
                new_stop = round(inv_price * 0.99, 2)
            log.info("inverse substitution: short %s -> buy %s (entry %.2f stop %.2f, %.2f%% stop)",
                     p.symbol, inv, inv_price, new_stop, stop_pct * 100)
            out.append(_Proposal(
                symbol=inv, entry_price=inv_price, stop_price=new_stop, side="buy",
                thesis=f"[inverse of short {p.symbol}] {p.thesis}",
            ))
        return out

    def _latest_price(self, symbol: str) -> float | None:
        """Best-effort latest price for a symbol via the short-term client."""
        try:
            raw = self.short_term.lookup_ticker(symbol, interval="5m", period="1d")
        except Exception:
            log.debug("lookup_ticker failed for %s", symbol, exc_info=True)
            return None
        snap = _summarize_quote(symbol, raw)
        price = snap.get("price")
        try:
            return float(price) if price is not None else None
        except (TypeError, ValueError):
            return None

    def _validate_and_size(self, proposals: list[_Proposal]) -> list[Decision]:
        from src.config import get_settings
        s = get_settings()
        acct = self.broker.get_account(self.sub_account)
        positions = self.broker.list_positions(self.sub_account)
        open_syms = {p.symbol for p in positions if p.qty > 0}
        # Existing per-symbol notional so the concentration cap counts what we
        # already hold — prevents accumulating one name past the cap over many
        # ticks (e.g. repeatedly topping up SOXS).
        held_notional = {p.symbol: abs(p.qty) * (p.mark_price or 0.0) for p in positions}
        # Only MATERIAL positions consume a concurrent-position slot; trivial
        # dust (< DUST_POSITION_PCT of equity) left over from tests/partial fills
        # must not block the agent from opening fresh names.
        dust_floor = acct.equity * DUST_POSITION_PCT if acct.equity else 0.0
        material_syms = {sym for sym, n in held_notional.items() if n >= dust_floor}
        out: list[Decision] = []
        slots_left = MAX_CONCURRENT_POSITIONS - len(material_syms)

        for p in proposals:
            d = Decision(symbol=p.symbol, side=p.side, qty=0.0, thesis=p.thesis)
            is_short = p.side == "sell"
            if slots_left <= 0 and p.symbol not in material_syms:
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
            # Bound notional so tight stops on cheap/volatile names can't consume
            # the account. Two independent limits, whichever is tighter wins:
            #  1. Agent risk discipline — never put more than DAY_MAX_POSITION_PCT
            #     of equity into one name per entry (always on; prevents the
            #     "74% of equity in one $4 ETF" failure mode).
            #  2. Optional venue caps — STOCK_REC_MAX_ORDER_USD / _MAX_SYMBOL_PCT
            #     (0 => disabled). These are the paper-broker's hard caps.
            if p.entry_price > 0:
                limits = [qty]
                # --- per-symbol concentration cap (position-aware) ---
                sym_pcts: list[float] = []
                day_pct = float(getattr(s, "day_max_position_pct", MAX_POSITION_PCT) or 0)
                if day_pct > 0:
                    sym_pcts.append(day_pct)
                try:
                    venue_pct = float(s.stock_rec_max_symbol_pct) / 100.0
                except (TypeError, ValueError):
                    venue_pct = 0.0
                if venue_pct > 0:
                    sym_pcts.append(venue_pct)
                if sym_pcts:
                    # Allowance is the cap MINUS what we already hold in this
                    # name, so topping up across ticks can't breach the cap.
                    cap_notional = acct.equity * min(sym_pcts)
                    remaining = cap_notional - held_notional.get(p.symbol, 0.0)
                    limits.append(math.floor(max(0.0, remaining) / p.entry_price))
                # --- per-order USD notional cap (venue leg only; 0 => disabled) ---
                try:
                    order_cap = float(s.stock_rec_max_order_usd)
                except (TypeError, ValueError):
                    order_cap = 0.0
                if order_cap > 0:
                    limits.append(math.floor(order_cap / p.entry_price))
                qty = min(limits)
            if qty <= 0:
                d.accepted = False
                d.reject_reason = "size_rounded_to_zero"
                out.append(d)
                continue
            d.qty = float(qty)
            d = validate(d, self.broker, self.sub_account)
            if d.accepted and p.symbol not in material_syms:
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


def _summarize_ideas(res: dict, *, held_pct: dict | None = None,
                     cap_pct: float = 20.0, context: dict | None = None) -> dict:
    """Compact, ranked view of upstream intraday ideas for the LLM.

    The upstream scanner already folds 9-source news sentiment into each idea's
    ``heat_score`` (via its ``news_spike`` catalyst signal, which shows up in
    ``signal_tags``). We surface the decision-relevant numbers — conviction
    (``heat_score``), tier, the fired signals, whether a news catalyst is among
    them, and the pre-computed entry/stop/target/RR/$-risk — sorted by
    conviction so the agent can rank and DIVERSIFY instead of hammering one name.

    Each idea is annotated with ``already_held`` / ``room_left`` so the LLM skips
    names that are already at their per-name cap and deploys idle cash elsewhere.
    """
    if not isinstance(res, dict) or res.get("error"):
        return {"error": (res or {}).get("error", "no_ideas")}
    held_pct = held_pct or {}
    rows = res.get("rows") or res.get("ideas") or []
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        tags = str(r.get("signal_tags") or "")
        sym = (r.get("ticker") or r.get("symbol") or "").upper()
        direction = str(r.get("direction") or "").lower()
        # The order actually placed may be substituted: a SHORT of a leveraged/
        # inverse ETF becomes a LONG of its inverse counterpart. Judge the cap
        # against the symbol we'd ACTUALLY end up holding, so we hide e.g.
        # "short SCO" when we already hold UCO at its cap.
        effective = sym
        if direction in ("short", "sell") and sym in INVERSE_ETF:
            effective = INVERSE_ETF[sym]
        cur = held_pct.get(effective, held_pct.get(sym, 0.0))
        out.append({
            "ticker": sym,
            "direction": r.get("direction"),
            "effective_symbol": effective,
            "tier": r.get("tier"),
            "heat_score": r.get("heat_score"),           # 0..100 conviction
            "signal_tags": tags,                          # scanners that fired
            "has_news_catalyst": "news_spike" in tags,    # news moved the score
            "earnings_soon": "earnings_soon" in tags,
            "entry": r.get("entry"),
            "stop": r.get("stop"),
            "target": r.get("target"),
            "rr": r.get("rr"),
            "dollar_risk": r.get("dollar_risk"),
            "dollar_gain": r.get("dollar_gain"),
            "already_held_pct": round(cur, 1),
            "room_left": round(max(0.0, cap_pct - cur), 1),   # % of equity still deployable
            "at_cap": cur >= cap_pct - 0.5,
        })
    out.sort(key=lambda x: (x.get("heat_score") or 0), reverse=True)
    # Hide names that are already at their per-name cap: proposing them just
    # rounds to zero shares and wastes the tick. Removing them forces the agent
    # to rotate into fresh names and deploy idle cash instead of re-proposing
    # the same maxed-out ticker every minute.
    tradable = [i for i in out if not i["at_cap"]]
    hidden = [i["ticker"] for i in out if i["at_cap"]]
    result = {"count": len(tradable), "ideas": tradable,
              "note": ("Ranked by heat_score (news sentiment is already folded in via "
                       "the news_spike signal; has_news_catalyst flags it). These are "
                       "names with ROOM LEFT — names already at the per-name "
                       f"~{cap_pct:.0f}% cap have been removed. DIVERSIFY: open NEW "
                       "positions across several uncorrelated ideas and deploy idle "
                       "cash; do not sit idle when tradable ideas exist.")}
    if hidden:
        result["hidden_at_cap"] = hidden
    if context:
        result["portfolio"] = context
    return result


def _summarize_news(symbol: str, raw: dict) -> dict:
    """Condense a get_news payload into headlines + aggregated sentiment.

    Accepts either the stock-recommender shape (``articles`` with per-article
    ``sentiment: {score,label}``) or the short-term shape (``articles`` with a
    flat ``sentiment`` float, or none). Produces a compact, decision-ready read:
    an average sentiment score, a label, bullish/bearish counts and the few
    most-impactful headlines.
    """
    if not isinstance(raw, dict) or raw.get("error"):
        return {"symbol": symbol.upper(), "error": (raw or {}).get("error", "no_data")}
    articles = raw.get("articles") or raw.get("news") or []
    if not articles:
        return {"symbol": symbol.upper(), "count": 0, "sentiment_label": "no_news",
                "note": "no recent articles"}

    scored: list[tuple[float, dict]] = []
    for a in articles:
        if not isinstance(a, dict):
            continue
        sent = a.get("sentiment")
        score = None
        if isinstance(sent, dict):
            score = _f(sent.get("score"))
        elif isinstance(sent, (int, float)):
            score = float(sent)
        elif a.get("sentiment_score") is not None:
            score = _f(a.get("sentiment_score"))
        scored.append((score if score is not None else 0.0, a))

    known = [s for s, _ in scored if s != 0.0]
    avg = round(sum(known) / len(known), 3) if known else 0.0
    bullish = sum(1 for s, _ in scored if s >= 0.08)
    bearish = sum(1 for s, _ in scored if s <= -0.08)
    label = "positive" if avg >= 0.08 else ("negative" if avg <= -0.08 else "neutral")

    def _headline(a: dict) -> str:
        return str(a.get("headline") or a.get("title") or a.get("summary") or "")[:160]

    top = sorted(scored, key=lambda x: abs(x[0]), reverse=True)[:5]
    top_headlines = [{"headline": _headline(a), "sentiment": round(s, 3),
                      "source": a.get("source")} for s, a in top if _headline(a)]

    return {
        "symbol": symbol.upper(),
        "count": len(articles),
        "sentiment_score": avg,          # −1 (very bearish) … +1 (very bullish)
        "sentiment_label": label,
        "bullish_articles": bullish,
        "bearish_articles": bearish,
        "top_headlines": top_headlines,
        "source": raw.get("source", "mcp"),
    }


def _summarize_analyst(symbol: str, raw: dict) -> dict:
    """Condense a lookup_ticker fundamentals payload into an analyst/sentiment read."""
    if not isinstance(raw, dict) or raw.get("error"):
        return {"symbol": symbol.upper(), "error": (raw or {}).get("error", "no_data")}
    price = _f(raw.get("price") or raw.get("last"))
    target = _f(raw.get("target_price"))
    upside = _f(raw.get("upside_pct"))
    if upside is None and price and target:
        upside = round(100.0 * (target - price) / price, 2)
    out = {
        "symbol": symbol.upper(),
        "price": price,
        "rating": raw.get("rating"),                 # Strong Buy … Strong Sell | N/A
        "target_price": target,
        "analyst_count": raw.get("analyst_count"),
        "upside_pct": upside,                          # implied upside to target
        "sentiment_score": _f(raw.get("sentiment_score")),
        "sentiment_label": raw.get("sentiment_label"),
        "source": raw.get("source", "mcp"),
    }
    # Drop keys the (local) fallback can't provide so the LLM sees a clean read.
    return {k: v for k, v in out.items() if v is not None}

