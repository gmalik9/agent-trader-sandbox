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

st.set_page_config(page_title="Agentic Trader", layout="wide")


@st.cache_resource
def _start_background_scheduler():
    """Boot APScheduler inside the Streamlit process.

    On Streamlit Cloud (and any single-process deployment) there is no separate
    scheduler. We start one here once per app instance. `@st.cache_resource`
    guarantees a single instance across reruns.

    Returns the runner so callers can introspect; failures are swallowed and
    logged so the UI still renders even if the scheduler can't start
    (e.g. missing secrets).
    """
    import logging
    logging.basicConfig(level=logging.INFO)
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
    st.caption("Scheduler: in-app" if _scheduler else "Scheduler: external / not running")


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
        st.metric("Day equity", f"${day_eq:,.2f}")
    with c2:
        long_eq = float(eq_long.iloc[-1]["equity"]) if not eq_long.empty else 0.0
        st.metric("Long equity", f"${long_eq:,.2f}")
    with c3:
        div = df(
            "SELECT COUNT(*) AS n FROM dual_divergence WHERE ts >= ?",
            ((datetime.now(timezone.utc) - pd.Timedelta(days=1)).isoformat(),),
        )
        st.metric("Divergence (24h)", int(div.iloc[0]["n"]) if not div.empty else 0)

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
        st.metric(f"{sub_account} cash (sandbox)", f"${cash:,.2f}")
    with snap_r:
        if aid_mirror:
            cash_m = dbm.get_cash(get_writable_conn(), aid_mirror)
            st.metric(f"{mirror} cash (alpaca)", f"${cash_m:,.2f}")

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

    if st.button(f"Tick {name} now", key=f"tick_{name}"):
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
                                value=int(runs.iloc[0]["id"]))
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
    new_kill = st.toggle("Kill switch (halts all agents)", value=cur_kill)
    if new_kill != cur_kill:
        dbm.set_setting(get_writable_conn(), "kill_switch", "on" if new_kill else "off")
        st.success(f"kill switch -> {'on' if new_kill else 'off'}")

    st.divider()
    st.subheader("Manual ticks")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Mark-to-market now"): _enqueue_tick("mtm")
    if c2.button("Day-trader tick"):     _enqueue_tick("day")
    if c3.button("Long-term tick"):      _enqueue_tick("long")
    if c4.button("Coordinator tick"):    _enqueue_tick("coordinator")

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
