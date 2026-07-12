"""Streamlit dashboard — read-only over SQLite.

The scheduler process owns all writes. This UI only writes to two tables:
  - `tick_requests`   (when user clicks a "Tick now" button)
  - `settings`        (when user changes the kill-switch / capital / split)
Everything else is a `SELECT` against the same SQLite file the scheduler
opened in WAL mode.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import db_path, get_settings
from src.sandbox import db as dbm
from src.analysis import pnl as pnl_mod

st.set_page_config(page_title="Agentic Trader", layout="wide")


@st.cache_resource
def _start_background_scheduler():
    """Boot APScheduler inside the Streamlit process.

    On Streamlit Cloud (and any single-process deployment) there is no separate
    scheduler. We start one here once per app instance. `@st.cache_resource`
    guarantees a single instance across reruns.

    Set `INPROCESS_SCHEDULER=0` to disable this — do that whenever a dedicated
    scheduler process is already running (e.g. `run.sh` or the Docker
    `scheduler` service), so the two don't double-tick the same book.

    Returns the runner so callers can introspect; failures are swallowed and
    logged so the UI still renders even if the scheduler can't start
    (e.g. missing secrets).
    """
    import logging
    import os
    logging.basicConfig(level=logging.INFO)
    if os.environ.get("INPROCESS_SCHEDULER", "1").strip().lower() in (
        "0", "false", "no", "off",
    ):
        logging.getLogger(__name__).info(
            "in-process scheduler disabled (INPROCESS_SCHEDULER); "
            "expecting an external scheduler process")
        return None
    try:
        from src.scheduler.runner import SchedulerRunner
        runner = SchedulerRunner(background=True)
        runner.start_background()
        return runner
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).exception("background scheduler failed: %s", e)
        return None


_scheduler = _start_background_scheduler()


@st.cache_resource
def get_writable_conn() -> sqlite3.Connection:
    """Single connection used for the rare UI writes (tick requests, settings)."""
    conn = dbm.get_conn(db_path())
    dbm.migrate(conn)
    return conn


def df(query: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_writable_conn()
    return pd.read_sql_query(query, conn, params=params)


def _aid(name: str) -> int | None:
    row = get_writable_conn().execute(
        "SELECT id FROM accounts WHERE name=?", (name,)
    ).fetchone()
    return row["id"] if row else None


def _enqueue_tick(agent: str) -> None:
    get_writable_conn().execute(
        "INSERT INTO tick_requests(ts, agent, requested_by) VALUES (?, ?, 'ui')",
        (datetime.now(timezone.utc).isoformat(), agent),
    )
    st.success(f"queued {agent} tick — scheduler will pick it up within ~5s")


def _scheduler_status() -> tuple[str, str]:
    """Report scheduler liveness for the header caption.

    Prefers the in-process scheduler flag; otherwise checks the heartbeat the
    external scheduler writes to `settings` every ~5s.
    """
    help_text = ("Where the trading scheduler runs. 'in-app' means this Streamlit "
                 "process ticks the agents. 'external' means a separate scheduler "
                 "process/container does (the recommended Docker setup). 'stale' "
                 "means no heartbeat was seen in the last 30s — start it with "
                 "`./trader start` or check `./trader logs scheduler`.")
    if _scheduler:
        return "Scheduler: in-app", help_text
    hb = dbm.get_setting(get_writable_conn(), "scheduler_heartbeat")
    if hb:
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(hb)).total_seconds()
            if age < 30:
                return "Scheduler: external ✓", help_text
            return f"Scheduler: external (stale {int(age)}s)", help_text
        except ValueError:
            pass
    return "Scheduler: not detected", help_text


def _pnl_by_symbol(account_id: int) -> pd.DataFrame:
    rows = pnl_mod.pnl_by_symbol(get_writable_conn(), account_id)
    cols = ["symbol", "realized_pnl", "unrealized_pnl", "fees", "net_pnl",
            "open_qty", "avg_cost", "mark", "trades"]
    return pd.DataFrame(rows, columns=cols)


def _realized_pnl_timeseries(account_id: int) -> pd.DataFrame:
    rows = pnl_mod.realized_pnl_timeseries(get_writable_conn(), account_id)
    return pd.DataFrame(rows, columns=["ts", "cum_realized"])


def _render_pnl_analysis(account_id: int, label: str) -> None:
    """Render the P&L analysis block (metrics + per-stock table + chart)."""
    st.markdown("### P&L analysis")
    pnl = _pnl_by_symbol(account_id)
    if pnl.empty:
        st.info("No filled trades yet for this account, so there is no P&L to show. "
                "Trigger a tick (during market hours for the day trader) to generate fills.")
        return

    realized = float(pnl["realized_pnl"].sum())
    unrealized = float(pnl["unrealized_pnl"].sum())
    fees = float(pnl["fees"].sum())
    net = float(pnl["net_pnl"].sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Realized P&L", f"${realized:,.2f}",
              help="Locked-in profit/loss from closed (sold) quantity, using average-cost "
                   "accounting. Does not include fees.")
    m2.metric("Unrealized P&L", f"${unrealized:,.2f}",
              help="Paper profit/loss on positions still open, marked at the latest price. "
                   "Changes as prices move; not yet locked in.")
    m3.metric("Fees", f"${fees:,.2f}",
              help="Total commissions/slippage fees charged on this account's fills.")
    m4.metric("Net P&L", f"${net:,.2f}",
              help="Realized + unrealized − fees. The bottom-line total for this account.")

    st.write("**Per-stock P&L**")
    st.dataframe(
        pnl, use_container_width=True, hide_index=True,
        column_config={
            "symbol": st.column_config.TextColumn("Symbol", help="Ticker."),
            "realized_pnl": st.column_config.NumberColumn(
                "Realized", format="$%.2f", help="Locked-in P&L from sold quantity."),
            "unrealized_pnl": st.column_config.NumberColumn(
                "Unrealized", format="$%.2f", help="Paper P&L on the still-open quantity."),
            "fees": st.column_config.NumberColumn(
                "Fees", format="$%.2f", help="Fees charged on this symbol's fills."),
            "net_pnl": st.column_config.NumberColumn(
                "Net", format="$%.2f", help="Realized + unrealized − fees for this symbol."),
            "open_qty": st.column_config.NumberColumn(
                "Open qty", help="Shares still held (0 if fully closed)."),
            "avg_cost": st.column_config.NumberColumn(
                "Avg cost", format="$%.2f", help="Average cost basis of the open shares."),
            "mark": st.column_config.NumberColumn(
                "Mark", format="$%.2f", help="Latest price used to mark open shares."),
            "trades": st.column_config.NumberColumn(
                "Fills", help="Number of filled orders for this symbol."),
        },
    )

    ts = _realized_pnl_timeseries(account_id)
    if not ts.empty:
        st.write("**Cumulative realized P&L (net of fees)**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(ts["ts"]), y=ts["cum_realized"],
            mode="lines+markers", name=label, line={"width": 2}))
        fig.update_layout(height=280, margin={"t": 10, "b": 20, "l": 20, "r": 20},
                           yaxis_title="USD")
        st.plotly_chart(fig, use_container_width=True)


# ---------------- header ----------------

s = get_settings()
st.title("Agentic Trader — Sandbox")
st.caption("Paper / simulated only. No live-money path exists.")

kill = dbm.get_setting(get_writable_conn(), "kill_switch") == "on"
hdr_l, hdr_r = st.columns([3, 1])
with hdr_l:
    st.markdown(f"**Backend:** `{s.broker_backend}`   "
                 f"**LLM:** `{s.llm_provider}/{s.llm_model}`   "
                 f"**Capital:** ${s.capital_total:,.0f} "
                 f"({s.split_day_pct:.0f}% day / {100 - s.split_day_pct:.0f}% long)")
with hdr_r:
    if kill:
        st.error("KILL SWITCH ON")
    else:
        st.success("Kill switch OFF")
    _sched_status, _sched_help = _scheduler_status()
    st.caption(_sched_status, help=_sched_help)


tabs = st.tabs(["Overview", "Day-Trader", "Long-Term", "History",
                  "Agent Runs", "Settings"])


# ---------------- Overview ----------------

with tabs[0]:
    st.subheader("Equity curves")
    eq_day = df(
        "SELECT ts, equity FROM equity_curve WHERE account_id=? ORDER BY ts",
        (_aid("day"),),
    )
    eq_long = df(
        "SELECT ts, equity FROM equity_curve WHERE account_id=? ORDER BY ts",
        (_aid("long"),),
    )

    fig = go.Figure()
    if not eq_day.empty:
        fig.add_trace(go.Scatter(x=pd.to_datetime(eq_day["ts"]), y=eq_day["equity"],
                                   name="Day (sandbox primary)"))
    if not eq_long.empty:
        fig.add_trace(go.Scatter(x=pd.to_datetime(eq_long["ts"]), y=eq_long["equity"],
                                   name="Long (sandbox primary)"))
    # Also overlay alpaca-mirror sub-accounts if they have any equity.
    for sub, label in (("day_alpaca", "Day (alpaca mirror)"),
                        ("long_alpaca", "Long (alpaca mirror)")):
        aid = _aid(sub)
        if aid is None:
            continue
        eq = df("SELECT ts, equity FROM equity_curve WHERE account_id=? ORDER BY ts", (aid,))
        if not eq.empty:
            fig.add_trace(go.Scatter(x=pd.to_datetime(eq["ts"]), y=eq["equity"],
                                       name=label, line={"dash": "dot"}))
    fig.update_layout(height=420, margin={"t": 20, "b": 20, "l": 20, "r": 20})
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        day_eq = float(eq_day.iloc[-1]["equity"]) if not eq_day.empty else 0.0
        st.metric("Day equity", f"${day_eq:,.2f}",
                   help="Latest total equity (cash + positions) of the day-trading "
                        "sub-account on the local sandbox venue. Updated by the "
                        "mark-to-market job every minute during market hours.")
    with c2:
        long_eq = float(eq_long.iloc[-1]["equity"]) if not eq_long.empty else 0.0
        st.metric("Long equity", f"${long_eq:,.2f}",
                   help="Latest total equity (cash + positions) of the long-term "
                        "sub-account on the local sandbox venue.")
    with c3:
        div = df(
            "SELECT COUNT(*) AS n FROM dual_divergence WHERE ts >= ?",
            ((datetime.now(timezone.utc) - pd.Timedelta(days=1)).isoformat(),),
        )
        st.metric("Divergence (24h)", int(div.iloc[0]["n"]) if not div.empty else 0,
                   help="Number of times in the last 24h the sandbox and Alpaca-paper "
                        "legs disagreed (fill price gap, a failed secondary order, or a "
                        "status mismatch). See the History tab's divergence log for detail.")

    st.subheader("Recent orders (both venues)")
    orders = df(
        "SELECT ts, venue, agent, symbol, side, qty, status, fill_price, fees, dual_group_id "
        "FROM orders ORDER BY ts DESC LIMIT 50"
    )
    st.dataframe(orders, use_container_width=True, hide_index=True)


def _agent_tab(name: str, sub_account: str, mirror: str):
    aid_primary = _aid(sub_account)
    aid_mirror = _aid(mirror)
    snap_l, snap_r = st.columns(2)
    with snap_l:
        cash = dbm.get_cash(get_writable_conn(), aid_primary) if aid_primary else 0
        st.metric(f"{sub_account} cash (sandbox)", f"${cash:,.2f}",
                   help="Uninvested cash in this sub-account on the local sandbox venue. "
                        "Buys reduce it; sells increase it.")
    with snap_r:
        if aid_mirror:
            cash_m = dbm.get_cash(get_writable_conn(), aid_mirror)
            st.metric(f"{mirror} cash (alpaca)", f"${cash_m:,.2f}",
                       help="Cash reported for the mirrored Alpaca paper account, when the "
                            "dual broker is active. Blank if only the sandbox is running.")

    positions = df(
        "SELECT symbol, qty, avg_cost, mark_price FROM positions_snapshot ps "
        "WHERE account_id=? AND ts=(SELECT MAX(ts) FROM positions_snapshot WHERE account_id=ps.account_id AND symbol=ps.symbol) "
        "ORDER BY symbol", (aid_primary,),
    )
    st.write("**Current positions (primary)**")
    st.dataframe(positions, use_container_width=True, hide_index=True)

    st.write("**Recent orders**")
    o = df("SELECT ts, venue, symbol, side, qty, status, fill_price, fees FROM orders "
            "WHERE account_id IN (?, ?) ORDER BY ts DESC LIMIT 50",
            (aid_primary, aid_mirror or -1))
    st.dataframe(o, use_container_width=True, hide_index=True)

    st.divider()
    if aid_primary:
        _render_pnl_analysis(aid_primary, f"{sub_account} (sandbox)")

    if st.button(f"Tick {name} now", key=f"tick_{name}",
                  help=f"Queue one immediate {name} agent run. The scheduler picks it up "
                       "within ~5s, asks the LLM for trade proposals, and places any that "
                       "pass the risk checks. The day agent only trades during market hours; "
                       "the long agent trades whenever run."):
        _enqueue_tick(name)


with tabs[1]:
    st.subheader("Day-Trader")
    _agent_tab("day", "day", "day_alpaca")

with tabs[2]:
    st.subheader("Long-Term Investor")
    _agent_tab("long", "long", "long_alpaca")


# ---------------- History (dual grouping) ----------------

with tabs[3]:
    st.subheader("Dual execution history")
    paired = df(
        "SELECT dual_group_id, "
        "       MIN(ts) AS ts, "
        "       MAX(CASE WHEN venue='sandbox' THEN fill_price END) AS sandbox_fill, "
        "       MAX(CASE WHEN venue='alpaca_paper' THEN fill_price END) AS alpaca_fill, "
        "       MAX(symbol) AS symbol, MAX(side) AS side, MAX(qty) AS qty "
        "FROM orders WHERE dual_group_id IS NOT NULL "
        "GROUP BY dual_group_id ORDER BY ts DESC LIMIT 100"
    )
    if not paired.empty:
        paired["slippage_bps"] = (
            (paired["alpaca_fill"] - paired["sandbox_fill"]) / paired["sandbox_fill"] * 10_000
        ).round(2)
    st.dataframe(paired, use_container_width=True, hide_index=True)

    st.subheader("Divergence log")
    divs = df("SELECT ts, dual_group_id, kind, primary_val, secondary_val, note "
               "FROM dual_divergence ORDER BY ts DESC LIMIT 100")
    st.dataframe(divs, use_container_width=True, hide_index=True)


# ---------------- Agent Runs ----------------

with tabs[4]:
    st.subheader("Agent runs")
    runs = df(
        "SELECT id, ts, agent, status, latency_ms, error FROM agent_runs "
        "ORDER BY ts DESC LIMIT 100"
    )
    st.dataframe(runs, use_container_width=True, hide_index=True)
    if not runs.empty:
        sel = st.number_input("Inspect run id", min_value=int(runs["id"].min()),
                                max_value=int(runs["id"].max()),
                                value=int(runs.iloc[0]["id"]),
                                help="Enter an agent-run id from the table above to see its "
                                     "full prompt, the LLM's final text, every tool call it "
                                     "made, the sized/validated decisions, and any error.")
        detail = df("SELECT prompt, response, tools_called, decisions, error "
                     "FROM agent_runs WHERE id=?", (int(sel),))
        if not detail.empty:
            row = detail.iloc[0]
            st.write("**Prompt**")
            st.code(row["prompt"] or "")
            st.write("**Final text**")
            st.code(row["response"] or "")
            for col in ("tools_called", "decisions"):
                st.write(f"**{col}**")
                try:
                    st.json(json.loads(row[col]) if row[col] else {})
                except Exception:
                    st.code(row[col] or "")
            if row["error"]:
                st.error(row["error"])


# ---------------- Settings ----------------

with tabs[5]:
    st.subheader("Runtime controls")
    cur_kill = dbm.get_setting(get_writable_conn(), "kill_switch") == "on"
    new_kill = st.toggle("Kill switch (halts all agents)", value=cur_kill,
                          help="Master stop. When ON, every agent run short-circuits to "
                               "'halted' within one scheduler tick (≤5s) and places no "
                               "orders. Existing positions are left untouched. Turn OFF to "
                               "resume automated trading.")
    if new_kill != cur_kill:
        dbm.set_setting(get_writable_conn(), "kill_switch", "on" if new_kill else "off")
        st.success(f"kill switch -> {'on' if new_kill else 'off'}")

    st.divider()
    st.subheader("Manual ticks")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Mark-to-market now",
                  help="Revalue all open positions in both sub-accounts at the latest "
                       "prices and append a point to the equity curves. Safe to run "
                       "anytime; places no orders."):
        _enqueue_tick("mtm")
    if c2.button("Day-trader tick",
                  help="Run the intraday day-trading agent once. Only opens trades during "
                       "market hours; force-flattens positions near the close."):
        _enqueue_tick("day")
    if c3.button("Long-term tick",
                  help="Run the long-horizon investor agent once. Reviews recommendations "
                       "and rebalances toward target weights; trades any time it is run."):
        _enqueue_tick("long")
    if c4.button("Coordinator tick",
                  help="Run the capital coordinator once. Rebalances the cash split between "
                       "the day and long sub-accounts; normally acts only early in the month."):
        _enqueue_tick("coordinator")

    st.divider()
    st.subheader("Active configuration (read-only)")
    st.json({
        "broker_backend": s.broker_backend,
        "llm_provider": s.llm_provider,
        "llm_model": s.llm_model,
        "capital_total": s.capital_total,
        "split_day_pct": s.split_day_pct,
        "short_term_trader_path": s.short_term_trader_path,
        "stock_recommender_path": s.stock_recommender_path,
        "db": str(db_path()),
    })
