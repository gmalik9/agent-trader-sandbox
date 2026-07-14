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
from src.ui import theme as rh_theme

st.set_page_config(page_title="Agentic Trader", layout="wide",
                   page_icon="📈", initial_sidebar_state="collapsed")

# Robinhood-style theming. The theme value is driven by a toggle in the header;
# we read it here (top of the script) so the CSS repaints in the same rerun the
# toggle changes it. Pre-seed session state so the widget starts on it without a
# default/session_state conflict warning.
if "rh_theme" not in st.session_state:
    st.session_state["rh_theme"] = "dark"
_THEME = st.session_state["rh_theme"]
st.markdown(rh_theme.inject_css(_THEME), unsafe_allow_html=True)


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


# ---------------- readable time helpers ----------------

def _parse_ts(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _fmt_ts(iso: str | None) -> str:
    """Human-readable UTC timestamp, e.g. 'Mon, Jul 13 2026 · 4:57 PM UTC'."""
    dt = _parse_ts(iso)
    if dt is None:
        return "—"
    return dt.strftime("%a, %b %-d %Y · %-I:%M %p UTC")


def _humanize_age(iso: str | None) -> str:
    """Relative age, e.g. '2 min ago', '3 hr ago', 'just now'."""
    dt = _parse_ts(iso)
    if dt is None:
        return "never"
    secs = (datetime.now(timezone.utc) - dt).total_seconds()
    if secs < 0:
        return "just now"
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} hr ago"
    return f"{int(secs // 86400)} day(s) ago"


def _readable_ts_column(frame: pd.DataFrame, col: str = "ts", new: str = "When") -> pd.DataFrame:
    """Return a copy with the ISO `col` replaced by a readable `new` column up front."""
    if frame.empty or col not in frame.columns:
        return frame
    out = frame.copy()
    out.insert(0, new, out[col].map(_fmt_ts))
    out = out.drop(columns=[col])
    return out


def _to_dt(series: pd.Series) -> pd.Series:
    """Robustly parse ISO-8601 timestamps that may mix microsecond precision.

    Demo-seeded rows have no microseconds while live fills do; a fixed inferred
    format then fails, so we parse each element per ISO-8601.
    """
    return pd.to_datetime(series, format="ISO8601", utc=True)


@st.cache_data(ttl=30)
def _alpaca_account() -> dict | None:
    """Live Alpaca paper account (cash/equity/buying power), cached 30s.

    Uses the direct paper-only REST client so the UI process doesn't need the
    MCP subprocess. Returns None if credentials are missing or the call fails.
    """
    try:
        from src.brokers.alpaca_options import AlpacaOptions
        acct = AlpacaOptions().get_account()
        return {
            "cash": float(acct.get("cash", 0) or 0),
            "equity": float(acct.get("equity", 0) or 0),
            "buying_power": float(acct.get("buying_power", 0) or 0),
            "options_buying_power": float(acct.get("options_buying_power", 0) or 0),
            "last_equity": float(acct.get("last_equity", 0) or 0),
            "account_number": acct.get("account_number", ""),
            "status": acct.get("status", ""),
        }
    except Exception:
        return None


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


def _market_is_open() -> bool:
    try:
        from src.sandbox.clock import is_market_open
        return bool(is_market_open())
    except Exception:
        return False


def _agent_activity(agent: str) -> dict:
    """Recent-activity summary for an agent, for the status panel."""
    conn = get_writable_conn()
    last = conn.execute(
        "SELECT ts, status FROM agent_runs WHERE agent=? ORDER BY id DESC LIMIT 1",
        (agent,),
    ).fetchone()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    runs_today = conn.execute(
        "SELECT COUNT(*) n FROM agent_runs WHERE agent=? AND ts LIKE ?",
        (agent, f"{today}%"),
    ).fetchone()["n"]
    # Trades placed today = filled or routed orders from this agent today.
    trades_today = conn.execute(
        "SELECT COUNT(*) n FROM orders WHERE agent=? AND ts LIKE ? "
        "AND status IN ('filled','routed_external')",
        (agent, f"{today}%"),
    ).fetchone()["n"]
    last_ts = last["ts"] if last else None
    last_status = last["status"] if last else None
    # "Active" = a run in the last 15 minutes.
    dt = _parse_ts(last_ts)
    active = dt is not None and (datetime.now(timezone.utc) - dt).total_seconds() < 900
    return {
        "last_ts": last_ts,
        "last_status": last_status,
        "runs_today": runs_today,
        "trades_today": trades_today,
        "active": active,
    }


def _render_activity_panel() -> None:
    """Top-of-page 'is the agent working?' status strip."""
    mkt = _market_is_open()
    cols = st.columns([1.2, 1.4, 1.4])
    with cols[0]:
        if mkt:
            st.success("● Market OPEN", icon="🟢")
        else:
            st.warning("● Market CLOSED", icon="🔴")
        st.caption("The day-trader only opens trades while the market is open. "
                    "The long-term agent can trade any time it runs.")
    for col, agent, title in ((cols[1], "day", "Day-Trader"),
                               (cols[2], "long", "Long-Term")):
        a = _agent_activity(agent)
        with col:
            dot = "🟢" if a["active"] else "⚪"
            state = "running" if a["active"] else "idle"
            st.markdown(f"**{dot} {title} — {state}**")
            st.caption(
                f"Last run: {_humanize_age(a['last_ts'])}"
                + (f" ({a['last_status']})" if a['last_status'] else "")
                + f" · {a['runs_today']} runs today · "
                f"{a['trades_today']} trade(s) placed today"
            )


def _pnl_by_symbol(account_id: int) -> pd.DataFrame:
    rows = pnl_mod.pnl_by_symbol(get_writable_conn(), account_id)
    cols = ["symbol", "status", "realized_pnl", "unrealized_pnl", "fees",
            "net_pnl", "pnl_pct", "open_qty", "avg_cost", "mark",
            "cost_basis", "market_value", "trades"]
    return pd.DataFrame(rows, columns=cols)


def _realized_pnl_timeseries(account_id: int) -> pd.DataFrame:
    rows = pnl_mod.realized_pnl_timeseries(get_writable_conn(), account_id)
    return pd.DataFrame(rows, columns=["ts", "cum_realized"])


def _latest_reason_by_symbol(account_ids: tuple[int, ...]) -> dict[str, dict]:
    """Most recent order thesis + timestamp per symbol for the given accounts.

    Powers the 'why did it trade this' column next to each stock. Strips the
    internal 'reject:'/'alpaca_not_placed:' prefixes so the human-readable
    thesis shows.
    """
    if not account_ids:
        return {}
    placeholders = ",".join("?" for _ in account_ids)
    rows = get_writable_conn().execute(
        f"SELECT symbol, thesis, ts FROM orders WHERE account_id IN ({placeholders}) "
        "AND thesis IS NOT NULL ORDER BY id DESC",
        account_ids,
    ).fetchall()
    out: dict[str, dict] = {}
    for r in rows:
        sym = r["symbol"]
        if sym in out:
            continue
        thesis = r["thesis"] or ""
        if "|" in thesis:  # drop internal reason prefix, keep the human thesis
            thesis = thesis.split("|", 1)[1] or thesis
        out[sym] = {"reason": thesis.strip(), "ts": r["ts"]}
    return out


def _render_pnl_analysis(account_id: int, label: str, *,
                         reason_account_ids: tuple[int, ...] = ()) -> None:
    """Render the P&L analysis block (metrics + per-stock table + chart)."""
    st.markdown("### P&L analysis")
    pnl = _pnl_by_symbol(account_id)
    if pnl.empty:
        st.info("No filled trades yet for this account, so there is no P&L to show. "
                "Trigger a tick (during market hours for the day trader) to generate fills.")
        return

    # Attach the agent's reasoning + last-traded timestamp per stock.
    reasons = _latest_reason_by_symbol(reason_account_ids or (account_id,))
    pnl["reason"] = pnl["symbol"].map(lambda s: reasons.get(s, {}).get("reason", ""))
    pnl["last_traded"] = pnl["symbol"].map(lambda s: _fmt_ts(reasons.get(s, {}).get("ts")))

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

    st.write("**Trades & holdings — per stock**")
    st.caption("What the agent bought, whether it is still holding or has sold, "
                "at what price, the resulting profit or loss, and *why* it traded "
                "(its reasoning), with the last-traded time.")
    st.dataframe(
        pnl, use_container_width=True, hide_index=True,
        column_config={
            "symbol": st.column_config.TextColumn("Symbol", help="Ticker."),
            "status": st.column_config.TextColumn(
                "Status", help="'Long'/'Short' = still open, '...(partly ...)' = partly "
                               "closed, 'Closed' = fully exited."),
            "open_qty": st.column_config.NumberColumn(
                "Shares held", help="Signed shares still held (negative = short)."),
            "avg_cost": st.column_config.NumberColumn(
                "Entry price (avg)", format="$%.2f",
                help="Average price the agent entered the open position at."),
            "mark": st.column_config.NumberColumn(
                "Current price", format="$%.2f",
                help="Latest market price used to value open shares."),
            "cost_basis": st.column_config.NumberColumn(
                "Invested", format="$%.2f", help="Money committed to the open position."),
            "market_value": st.column_config.NumberColumn(
                "Current value", format="$%.2f", help="Open shares × current price."),
            "unrealized_pnl": st.column_config.NumberColumn(
                "Unrealized P&L", format="$%.2f",
                help="Paper profit/loss on the position still held."),
            "pnl_pct": st.column_config.NumberColumn(
                "P&L %", format="%.2f%%", help="Unrealized P&L as a % of money invested."),
            "realized_pnl": st.column_config.NumberColumn(
                "Realized P&L", format="$%.2f", help="Locked-in P&L from shares already closed."),
            "fees": st.column_config.NumberColumn(
                "Fees", format="$%.2f", help="Fees charged on this symbol's fills."),
            "net_pnl": st.column_config.NumberColumn(
                "Net P&L", format="$%.2f", help="Realized + unrealized − fees for this symbol."),
            "reason": st.column_config.TextColumn(
                "Why (agent reasoning)", width="large",
                help="The agent's most recent thesis for trading this stock."),
            "last_traded": st.column_config.TextColumn(
                "Last traded", help="When the agent last placed an order in this stock."),
            "trades": st.column_config.NumberColumn(
                "Fills", help="Number of filled orders for this symbol."),
        },
        column_order=["symbol", "status", "open_qty", "avg_cost", "mark",
                       "cost_basis", "market_value", "unrealized_pnl", "pnl_pct",
                       "realized_pnl", "fees", "net_pnl", "reason", "last_traded", "trades"],
    )

    ts = _realized_pnl_timeseries(account_id)
    if not ts.empty:
        st.write("**Cumulative realized P&L (net of fees)**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=_to_dt(ts["ts"]), y=ts["cum_realized"],
            mode="lines+markers", name=label, line={"width": 2}))
        fig.update_layout(height=280, margin={"t": 10, "b": 20, "l": 20, "r": 20},
                           yaxis_title="USD")
        _last = float(ts["cum_realized"].iloc[-1]) if not ts.empty else 0.0
        rh_theme.style_fig(fig, _THEME, positive=_last >= 0)
        st.plotly_chart(fig, use_container_width=True)


def _render_reasoning_card(exp: dict) -> None:
    """Render one agent run as a readable 'why did it trade' card."""
    icon = {"ok": "✅", "no-op": "➖", "halted": "🛑", "error": "⚠️"}.get(exp["status"], "•")
    title = (f"{icon} {exp['agent'].title()} agent · {_fmt_ts(exp['ts'])} "
             f"· {_humanize_age(exp['ts'])}")
    with st.expander(title, expanded=False):
        if exp["rationale"]:
            st.markdown("**Agent's rationale**")
            st.info(exp["rationale"])
        else:
            st.caption("No natural-language rationale recorded for this run.")

        if exp["decisions"]:
            st.markdown("**Decisions** — what it chose to do and why")
            dec = pd.DataFrame(exp["decisions"])
            st.dataframe(
                dec, use_container_width=True, hide_index=True,
                column_config={
                    "symbol": st.column_config.TextColumn("Symbol"),
                    "action": st.column_config.TextColumn("Action", help="Buy or Sell."),
                    "qty": st.column_config.NumberColumn("Qty"),
                    "thesis": st.column_config.TextColumn(
                        "Why (thesis)", help="The agent's stated reason for this trade.",
                        width="large"),
                    "outcome": st.column_config.TextColumn(
                        "Outcome", help="Whether the trade was placed or skipped by a "
                                        "risk rule (and why)."),
                    "accepted": None,
                },
                column_order=["symbol", "action", "qty", "thesis", "outcome"],
            )
        else:
            st.caption("No trades were proposed in this run.")

        if exp["data_sources"]:
            st.markdown("**Data sources considered** — citations behind the decision")
            src = pd.DataFrame(exp["data_sources"])
            st.dataframe(
                src, use_container_width=True, hide_index=True,
                column_config={
                    "label": st.column_config.TextColumn(
                        "Source", help="Which upstream data feed the agent queried."),
                    "query": st.column_config.TextColumn(
                        "Query", help="The exact parameters it requested."),
                    "summary": st.column_config.TextColumn(
                        "What it returned", help="A summary of the data it received and "
                                                 "reasoned over.", width="large"),
                    "tool": None,
                },
                column_order=["label", "query", "summary"],
            )
        else:
            st.caption("No upstream data-source calls recorded for this run.")

        if exp["error"]:
            st.error(f"Error: {exp['error']}")


def _render_agent_reasoning(agent: str, *, latest_only: bool = False, limit: int = 10) -> None:
    """Render readable reasoning cards for an agent's recent runs."""
    from src.analysis import reasoning as R
    n = 1 if latest_only else limit
    rows = get_writable_conn().execute(
        "SELECT id, ts, agent, status, response, decisions, tools_called, error "
        "FROM agent_runs WHERE agent=? ORDER BY id DESC LIMIT ?",
        (agent, n),
    ).fetchall()
    st.markdown("### Agent reasoning" if not latest_only else "### Latest agent reasoning")
    st.caption("Why the agent bought, held, or sold — its rationale, each decision's "
                "thesis, and the exact data sources it cited.")
    if not rows:
        st.info("This agent has not run yet. Trigger a tick to generate reasoning.")
        return
    for row in rows:
        _render_reasoning_card(R.explain_run(row))


# ---------------- header ----------------

s = get_settings()
_pal = rh_theme.palette(_THEME)

# Top bar: brand on the left, theme toggle + status on the right.
brand_l, brand_r = st.columns([3, 1.15])
with brand_l:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;'>"
        "<span style='font-size:1.7rem;font-weight:800;letter-spacing:-.03em;'>Agentic Trader</span>"
        f"<span style='color:{_pal['text_muted']};font-weight:600;font-size:.8rem;"
        "border:1px solid var(--rh-border);border-radius:999px;padding:2px 10px;'>PAPER</span>"
        "</div>",
        unsafe_allow_html=True)
with brand_r:
    st.segmented_control(
        "Theme", ["dark", "light"], key="rh_theme",
        format_func=lambda v: "🌙 Dark" if v == "dark" else "☀️ Light",
        label_visibility="collapsed")

kill = dbm.get_setting(get_writable_conn(), "kill_switch") == "on"
_acct = _alpaca_account()

# Robinhood-style hero: big portfolio value with today's $ / % change.
if _acct is not None:
    eq = _acct["equity"]
    prev = _acct["last_equity"] or eq
    chg = eq - prev
    chg_pct = (chg / prev * 100) if prev else 0.0
    up = chg >= 0
    col = _pal["green"] if up else _pal["red"]
    arrow = "▲" if up else "▼"
    st.markdown(
        f"<div style='margin:6px 0 2px;color:{_pal['text_muted']};font-weight:600;"
        "font-size:.8rem;'>Portfolio value · Alpaca paper</div>"
        f"<div style='font-size:2.6rem;font-weight:800;letter-spacing:-.03em;"
        f"line-height:1.1;'>${eq:,.2f}</div>"
        f"<div style='color:{col};font-weight:700;font-size:1rem;margin-top:2px;'>"
        f"{arrow} ${abs(chg):,.2f} ({chg_pct:+.2f}%) <span style='color:{_pal['text_muted']};"
        "font-weight:500;'>Today</span></div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 4px;'>"
        + "".join(
            f"<span style='background:{_pal['surface']};border:1px solid var(--rh-border);"
            f"border-radius:999px;padding:5px 12px;font-size:.8rem;color:{_pal['text_muted']};'>"
            f"<b style='color:{_pal['text']};'>{lbl}</b> {val}</span>"
            for lbl, val in [
                ("Cash", f"${_acct['cash']:,.0f}"),
                ("Buying power", f"${_acct['buying_power']:,.0f}"),
                ("Options BP", f"${_acct['options_buying_power']:,.0f}"),
                ("Backend", s.broker_backend),
                ("LLM", s.llm_model),
                ("Account", f"{_acct['account_number']} ({_acct['status']})"),
            ])
        + "</div>",
        unsafe_allow_html=True)
else:
    st.caption("Alpaca paper account not reachable — set ALPACA_API_KEY_ID / "
                "ALPACA_SECRET_KEY / ALPACA_PAPER=true to pull live cash.")

# Status strip: kill switch + scheduler.
_sched_status, _sched_help = _scheduler_status()
if kill:
    st.error("🛑 KILL SWITCH ON — agents halted.")

# Live throttling notification: if the LLM was rate-limited recently, the agent
# skips ticks — surface that prominently so idleness is explained.
_throttled_at = dbm.get_setting(get_writable_conn(), "llm_throttled_at")
_throttle_dt = _parse_ts(_throttled_at)
if _throttle_dt is not None:
    _age_s = (datetime.now(timezone.utc) - _throttle_dt).total_seconds()
    if _age_s < 180:  # throttled within the last 3 minutes → still relevant
        _detail = dbm.get_setting(get_writable_conn(), "llm_throttle_detail") or ""
        _warn_col, _btn_col = st.columns([5, 1])
        with _warn_col:
            st.warning(
                f"⚠️ LLM throttled { _humanize_age(_throttled_at) } — the model provider "
                f"returned HTTP 429 (rate limit), so some ticks were skipped. "
                f"The agent retries with backoff and downshifts to a fallback model; "
                f"trading resumes automatically once quota frees up. "
                f"({_detail})",
                icon="⚠️")
        with _btn_col:
            # Manual re-trigger, in addition to the periodic automatic tick, so
            # the user can retry immediately once quota may have freed up.
            if st.button("↻ Retry now", key="retry_throttled_day",
                          type="primary", use_container_width=True,
                          help="Queue an immediate day-agent run now instead of waiting "
                               "for the next automatic tick. Useful right after a throttle "
                               "to retry once the rate limit may have cleared."):
                _enqueue_tick("day")

st.markdown(
    f"<div style='color:{_pal['text_muted']};font-size:.82rem;margin:6px 0 2px;'>"
    f"{'🔴 Kill switch ON' if kill else '🟢 Kill switch OFF'} &nbsp;·&nbsp; {_sched_status}</div>",
    unsafe_allow_html=True)
st.divider()


tabs = st.tabs(["Overview", "Day-Trader", "Long-Term", "History",
                  "Agent Runs", "Settings"])


# ---------------- Overview ----------------

with tabs[0]:
    st.subheader("Agent activity")
    _render_activity_panel()
    st.divider()
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
        fig.add_trace(go.Scatter(x=_to_dt(eq_day["ts"]), y=eq_day["equity"],
                                   name="Day (sandbox primary)"))
    if not eq_long.empty:
        fig.add_trace(go.Scatter(x=_to_dt(eq_long["ts"]), y=eq_long["equity"],
                                   name="Long (sandbox primary)"))
    # Also overlay alpaca-mirror sub-accounts if they have any equity.
    for sub, label in (("day_alpaca", "Day (alpaca mirror)"),
                        ("long_alpaca", "Long (alpaca mirror)")):
        aid = _aid(sub)
        if aid is None:
            continue
        eq = df("SELECT ts, equity FROM equity_curve WHERE account_id=? ORDER BY ts", (aid,))
        if not eq.empty:
            fig.add_trace(go.Scatter(x=_to_dt(eq["ts"]), y=eq["equity"],
                                       name=label, line={"dash": "dot"}))
    fig.update_layout(height=420, margin={"t": 20, "b": 20, "l": 20, "r": 20})
    rh_theme.style_fig(fig, _THEME)
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
    # Money actually deployed per order = filled qty × fill price (0 for
    # unfilled/rejected rows so they don't inflate the totals).
    if not orders.empty:
        fp = pd.to_numeric(orders["fill_price"], errors="coerce")
        qt = pd.to_numeric(orders["qty"], errors="coerce")
        filled = orders["status"].astype(str).eq("filled")
        orders["spend"] = (fp * qt).where(filled, 0.0).fillna(0.0).round(2)
    st.dataframe(
        _readable_ts_column(orders), use_container_width=True, hide_index=True,
        column_config={
            "side": st.column_config.TextColumn("Side", help="buy or sell."),
            "fill_price": st.column_config.NumberColumn("Fill price", format="$%.2f"),
            "spend": st.column_config.NumberColumn(
                "Spend", format="$%.2f",
                help="Money deployed on this order (filled qty × fill price). "
                     "0 for unfilled or rejected orders."),
            "fees": st.column_config.NumberColumn("Fees", format="$%.2f"),
            "status": st.column_config.TextColumn(
                "Status", help="filled = executed on the sandbox; routed_external = sent "
                               "to Alpaca paper; rejected = blocked by a risk cap or no bar."),
        },
    )

    # Per-symbol spend summary (filled orders only), with a grand total. Buys add
    # exposure, sells reduce it, so we net them per symbol per venue.
    spend = df(
        "SELECT venue, symbol, "
        "SUM(CASE WHEN side='buy'  THEN qty*COALESCE(fill_price,0) ELSE 0 END) AS bought, "
        "SUM(CASE WHEN side='sell' THEN qty*COALESCE(fill_price,0) ELSE 0 END) AS sold, "
        "SUM(qty*COALESCE(fill_price,0)) AS gross, COUNT(*) AS fills "
        "FROM orders WHERE status='filled' GROUP BY venue, symbol "
        "ORDER BY gross DESC LIMIT 50"
    )
    st.subheader("Money spent per stock (filled)")
    if spend.empty:
        st.caption("No filled orders yet.")
    else:
        spend["net_exposure"] = (spend["bought"] - spend["sold"]).round(2)
        total_gross = float(spend["gross"].sum())
        equity_ref = None
        try:
            eq = df("SELECT equity FROM equity_curve WHERE account_id IN "
                    "(SELECT id FROM accounts WHERE venue='alpaca_paper') "
                    "ORDER BY ts DESC LIMIT 1")
            equity_ref = float(eq.iloc[0]["equity"]) if not eq.empty else None
        except Exception:
            equity_ref = None
        if equity_ref and equity_ref > 0:
            spend["pct_of_equity"] = (spend["net_exposure"] / equity_ref * 100).round(1)
        st.dataframe(
            spend, use_container_width=True, hide_index=True,
            column_config={
                "bought": st.column_config.NumberColumn("Bought", format="$%.2f"),
                "sold": st.column_config.NumberColumn("Sold", format="$%.2f"),
                "gross": st.column_config.NumberColumn(
                    "Gross traded", format="$%.2f",
                    help="Total notional traded (buys + sells) for this symbol."),
                "net_exposure": st.column_config.NumberColumn(
                    "Net exposure", format="$%.2f",
                    help="Buys − sells: approximate capital currently deployed."),
                "pct_of_equity": st.column_config.NumberColumn(
                    "% of equity", format="%.1f%%",
                    help="Net exposure as a share of live Alpaca equity — watch for "
                         "over-concentration in a single name."),
                "fills": st.column_config.NumberColumn("Fills"),
            },
        )
        c1, c2 = st.columns(2)
        c1.metric("Total gross traded (filled)", f"${total_gross:,.2f}")
        c2.metric("Total net exposure", f"${float(spend['net_exposure'].sum()):,.2f}")



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
        acct = _alpaca_account()
        if acct is not None:
            st.metric("Alpaca cash (live)", f"${acct['cash']:,.2f}",
                       help=f"Live cash on the Alpaca paper account {acct['account_number']} "
                            f"(equity ${acct['equity']:,.0f}, buying power "
                            f"${acct['buying_power']:,.0f}, options BP "
                            f"${acct['options_buying_power']:,.0f}). Shared across both "
                            "sub-accounts. Refreshes ~every 30s.")
        elif aid_mirror:
            cash_m = dbm.get_cash(get_writable_conn(), aid_mirror)
            st.metric(f"{mirror} cash (alpaca)", f"${cash_m:,.2f}",
                       help="Alpaca account unavailable — showing the local mirror ledger. "
                            "Set Alpaca paper keys to pull live cash.")

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
    st.dataframe(
        _readable_ts_column(o), use_container_width=True, hide_index=True,
        column_config={
            "side": st.column_config.TextColumn("Side"),
            "fill_price": st.column_config.NumberColumn("Fill price", format="$%.2f"),
            "fees": st.column_config.NumberColumn("Fees", format="$%.2f"),
        },
    )

    st.divider()
    if aid_primary:
        _render_pnl_analysis(aid_primary, f"{sub_account} (sandbox)",
                              reason_account_ids=tuple(
                                  a for a in (aid_primary, aid_mirror) if a))

    # Options (calls/puts) — routed straight to Alpaca, venue='alpaca_options'.
    opts = df(
        "SELECT ts, symbol, side, qty, status, fill_price, external_id, thesis "
        "FROM orders WHERE venue='alpaca_options' AND agent=? ORDER BY ts DESC LIMIT 30",
        (name,),
    )
    if not opts.empty:
        st.divider()
        st.markdown("### Options (calls/puts) — Alpaca")
        st.caption("Option orders route directly to Alpaca paper (OCC symbols, "
                    "1 contract = 100 shares). 'accepted'/'filled' with an external_id "
                    "means Alpaca took the order.")
        st.dataframe(
            _readable_ts_column(opts), use_container_width=True, hide_index=True,
            column_config={
                "symbol": st.column_config.TextColumn("Contract (OCC)"),
                "side": st.column_config.TextColumn("Side"),
                "qty": st.column_config.NumberColumn("Contracts"),
                "fill_price": st.column_config.NumberColumn("Fill", format="$%.2f"),
                "external_id": st.column_config.TextColumn("Alpaca ID"),
                "thesis": st.column_config.TextColumn("Why", width="large"),
            },
        )

    st.divider()
    _render_agent_reasoning(name, latest_only=True)

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
    st.dataframe(
        _readable_ts_column(paired), use_container_width=True, hide_index=True,
        column_config={
            "sandbox_fill": st.column_config.NumberColumn("Sandbox fill", format="$%.2f"),
            "alpaca_fill": st.column_config.NumberColumn("Alpaca fill", format="$%.2f"),
            "slippage_bps": st.column_config.NumberColumn(
                "Slippage (bps)", help="Difference between the Alpaca and sandbox fill "
                                       "prices, in basis points."),
        },
    )

    st.subheader("Divergence log")
    divs = df("SELECT ts, dual_group_id, kind, primary_val, secondary_val, note "
               "FROM dual_divergence ORDER BY ts DESC LIMIT 100")
    st.dataframe(_readable_ts_column(divs), use_container_width=True, hide_index=True)


# ---------------- Agent Runs ----------------

with tabs[4]:
    st.subheader("Agent runs & reasoning")
    st.caption("Every time an agent runs it records its rationale, the trades it "
                "decided on, and the exact data it consulted. Expand a card to read why.")

    fcol, ecol = st.columns([2, 1])
    with fcol:
        which = st.radio("Show", ["Both", "Day-Trader", "Long-Term"], horizontal=True,
                          help="Filter the reasoning cards by agent.")
    with ecol:
        if st.button("Export reasoning log",
                      help="Append all reasoning (decisions + cited data sources) to "
                           "data/reasoning_log.jsonl — a durable dataset for learning "
                           "from past trades."):
            try:
                from scripts.export_reasoning import export
                n = export()
                st.success(f"exported {n} new record(s) to data/reasoning_log.jsonl")
            except Exception as e:  # noqa: BLE001
                st.error(f"export failed: {e}")

    agent_filter = {"Day-Trader": "day", "Long-Term": "long"}.get(which)

    from src.analysis import reasoning as R
    if agent_filter:
        rows = get_writable_conn().execute(
            "SELECT id, ts, agent, status, response, decisions, tools_called, error "
            "FROM agent_runs WHERE agent=? ORDER BY id DESC LIMIT 30",
            (agent_filter,),
        ).fetchall()
    else:
        rows = get_writable_conn().execute(
            "SELECT id, ts, agent, status, response, decisions, tools_called, error "
            "FROM agent_runs ORDER BY id DESC LIMIT 30",
        ).fetchall()

    if not rows:
        st.info("No agent runs yet.")
    else:
        for row in rows:
            _render_reasoning_card(R.explain_run(row))


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
        "dual_primary": s.dual_primary,
        "llm_provider": s.llm_provider,
        "llm_model": s.llm_model,
        "capital_total": s.capital_total,
        "split_day_pct": s.split_day_pct,
        "allow_shorting": s.allow_shorting,
        "allow_leveraged": s.allow_leveraged,
        "max_leverage": s.max_leverage,
        "short_term_trader_path": s.short_term_trader_path,
        "stock_recommender_path": s.stock_recommender_path,
        "db": str(db_path()),
    })
    st.caption("Trading permissions and the dual-broker primary venue are set via "
                "environment / secrets (`ALLOW_SHORTING`, `ALLOW_LEVERAGED`, "
                "`MAX_LEVERAGE`, `DUAL_PRIMARY`). Paper/simulated only.")
