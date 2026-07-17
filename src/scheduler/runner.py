"""Scheduler runner — long-running process that owns all writes.

Jobs (UTC scheduling; market-hour gating is internal to each job):
- `mtm`        every 1 min — mark-to-market both sub-accounts
- `reconcile`  every 1 min — sync Alpaca order statuses into the local mirror
- `day_tick`   every 1 min — DayTraderAgent.run_once (high-frequency; adaptive
               model downshifts on throttling)
- `long_tick`  daily at 21:30 UTC (16:30 ET nominal)
- `coord_tick` daily at 21:45 UTC; acts only on the first trading week of the month
- `tick_poll`  every 5s — drains `tick_requests` from the Streamlit UI

A `data/scheduler.lock` file prevents double-start. SIGINT/SIGTERM trigger
a graceful shutdown.
"""

from __future__ import annotations

import logging
import os
import signal
import sqlite3
import threading
import time
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.agents.coordinator import Coordinator
from src.agents.day_trader import DayTraderAgent
from src.agents.long_term import LongTermAgent
from src.brokers.factory import build_broker
from src.config import DATA_DIR, db_path, get_settings
from src.llm.factory import get_provider
from src.mcp_clients.long_term import LongTermClient
from src.mcp_clients.short_term import ShortTermClient
from src.sandbox import db as dbm
from src.sandbox.clock import is_market_open, now_utc
from src.signals.local import (
    HybridShortTermClient,
    LocalLongTermClient,
    LocalShortTermClient,
)

log = logging.getLogger(__name__)
LOCK_PATH = DATA_DIR / "scheduler.lock"


class SchedulerRunner:
    def __init__(self, *, background: bool = False) -> None:
        self.settings = get_settings()
        self.conn: sqlite3.Connection | None = None
        self.short_term: ShortTermClient | LocalShortTermClient | None = None
        self.long_term: LongTermClient | LocalLongTermClient | None = None
        self.broker = None
        self.options = None
        self.provider = None
        self.background = background
        cls = BackgroundScheduler if background else BlockingScheduler
        self.scheduler = cls(
            timezone="UTC", job_defaults={"coalesce": True, "max_instances": 1,
                                            "misfire_grace_time": 60},
        )
        self._stop = threading.Event()
        # Monotonic timestamp of the in-flight day tick (None when idle). The
        # watchdog thread uses this to detect a hung tick that would otherwise
        # block every future tick via max_instances=1.
        self._day_tick_started_at: float | None = None

    # ---------------- lifecycle ----------------

    def _acquire_lock(self) -> None:
        # In containers the process manager (Docker/compose) already guarantees a
        # single scheduler instance, and the lock file lives on a mounted volume
        # that survives restarts — where a recycled PID can make a stale lock look
        # live. Allow skipping the pid-lock in that case.
        if os.environ.get("SCHEDULER_SKIP_LOCK", "").strip().lower() in (
            "1", "true", "yes", "on",
        ):
            return
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if LOCK_PATH.exists():
            try:
                pid = int(LOCK_PATH.read_text().strip())
            except (ValueError, OSError):
                pid = 0
            if pid and _pid_alive(pid):
                raise RuntimeError(f"scheduler already running (pid {pid}); "
                                    f"remove {LOCK_PATH} if stale")
        LOCK_PATH.write_text(str(os.getpid()))

    def _release_lock(self) -> None:
        try:
            if LOCK_PATH.exists() and LOCK_PATH.read_text().strip() == str(os.getpid()):
                LOCK_PATH.unlink()
        except OSError:
            pass

    def setup(self) -> None:
        self.conn = dbm.get_conn(db_path())
        dbm.migrate(self.conn)
        dbm.bootstrap_accounts(self.conn,
                                 capital_total=self.settings.capital_total,
                                 split_day_pct=self.settings.split_day_pct)

        # Idea/quote sources for the agents. Prefer the real sibling MCP servers
        # when their paths are configured and they launch; otherwise fall back to
        # the self-contained yfinance providers so the agents still trade (e.g.
        # on Streamlit Cloud, where the sibling repos don't exist).
        self.short_term = self._make_short_term()
        self.long_term, long_mcp = self._make_long_term()

        # The Alpaca paper leg REQUIRES the real upstream MCP client (it exposes
        # the write tools). Never wire it to the local fallback — pass only the
        # real MCP client (or None) so the broker falls back to sandbox-only.
        self.broker = build_broker(self.conn, long_term_client=long_mcp)

        # Options (calls/puts) go straight to Alpaca paper via a direct client,
        # enabled only when paper credentials are present and options are approved.
        self.options = self._make_options()

        try:
            self.provider = get_provider()
            log.info("LLM provider: %s / %s", self.provider.name, self.provider.model)
        except Exception as e:
            log.warning("no LLM provider configured (%s); LLM-driven ticks will error", e)
            self.provider = None

    def _make_short_term(self):
        """Real short-term MCP client (wrapped with a local yfinance fallback for
        ideas), or the pure local provider if the MCP can't start."""
        if self.settings.short_term_trader_path:
            try:
                real = ShortTermClient()
                real.start()
                log.info("short-term MCP client up (hybrid: local yfinance idea fallback)")
                return HybridShortTermClient(real)
            except Exception as e:
                log.warning("short-term MCP client failed to start (%s); "
                             "using local yfinance idea provider", e)
        else:
            log.info("SHORT_TERM_TRADER_PATH unset; using local yfinance idea provider")
        return LocalShortTermClient()

    def _make_long_term(self) -> tuple[LongTermClient | LocalLongTermClient,
                                        LongTermClient | None]:
        """Return (agent_client, alpaca_leg_client).

        ``agent_client`` is always usable (real MCP or local fallback).
        ``alpaca_leg_client`` is the real MCP client or None — the Alpaca paper
        broker leg is only wired when the real upstream is available.
        """
        if self.settings.stock_recommender_path:
            try:
                client = LongTermClient()
                client.start()
                log.info("long-term MCP client up")
                return client, client
            except Exception as e:
                log.warning("long-term MCP client failed to start (%s); "
                             "using local yfinance recommendation provider "
                             "(Alpaca mirror disabled)", e)
        else:
            log.info("STOCK_RECOMMENDER_PATH unset; using local yfinance "
                      "recommendation provider (Alpaca mirror disabled)")
        return LocalLongTermClient(), None

    def _make_options(self):
        """Return an AlpacaOptions client if paper options are available, else None."""
        try:
            from src.brokers.alpaca_options import AlpacaOptions
            opt = AlpacaOptions()
            if opt.options_enabled():
                log.info("Alpaca options enabled (calls/puts available)")
                return opt
            log.info("Alpaca options not approved on this account; options disabled")
        except Exception as e:  # noqa: BLE001
            log.warning("options client unavailable (%s); options disabled", e)
        return None

    def teardown(self) -> None:
        self._stop.set()
        for c in (self.short_term, self.long_term):
            if c is not None:
                try:
                    c.stop()
                except Exception:
                    log.exception("error stopping MCP client")
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
        self._release_lock()

    # ---------------- agents ----------------

    def _day_agent(self) -> DayTraderAgent:
        return DayTraderAgent(self.conn, self.broker, self.short_term,
                                provider=self.provider, options=self.options,
                                long_term=self.long_term)

    def _long_agent(self) -> LongTermAgent:
        return LongTermAgent(self.conn, self.broker, self.long_term,
                              provider=self.provider)

    def _coordinator(self) -> Coordinator:
        return Coordinator(self.conn, self.broker)

    # ---------------- jobs ----------------

    def job_mtm(self) -> None:
        now = now_utc()
        if not is_market_open(now):
            return
        try:
            for sub in ("day", "long"):
                self.broker.mark_to_market(now, sub_account=sub)
        except Exception:
            log.exception("mtm failed")

    def job_reconcile(self) -> None:
        # Sync Alpaca order statuses into the local mirror so fills show up.
        # Best-effort; runs regardless of market hours (fills can settle after
        # close). No-op for sandbox-only backends (broker has no `reconcile`).
        fn = getattr(self.broker, "reconcile", None)
        if not callable(fn):
            return
        try:
            fn()
        except Exception:
            log.exception("reconcile failed")

    def job_day_tick(self) -> None:
        if not is_market_open(now_utc()):
            return
        if not self._autopilot_enabled():
            # COPILOT-DRIVEN mode: don't make any LLM/PAT calls here — Copilot
            # does the reasoning via the autonomous-day-trader skill. Stops,
            # scans, reconcile and MTM keep running via their own jobs.
            return
        self._day_tick_started_at = time.monotonic()
        try:
            self._day_agent().run_once()
        except Exception:
            log.exception("day_tick failed")
        finally:
            self._day_tick_started_at = None

    def _autopilot_enabled(self) -> bool:
        """Whether the scheduler should run the LLM day_tick itself.

        The live `day_autopilot` setting ('on'/'off') wins if present; otherwise
        fall back to the DAY_AUTOPILOT config default. Set 'off' to hand the
        reasoning to Copilot (no PAT/LLM calls from the scheduler)."""
        try:
            val = dbm.get_setting(self.conn, "day_autopilot")
        except Exception:
            val = None
        if val is not None:
            return str(val).strip().lower() in ("1", "true", "yes", "on")
        return bool(getattr(self.settings, "day_autopilot", True))

    def job_stop_monitor(self) -> None:
        # Cheap, LLM-free guard: enforce each open day-position's stop/target
        # between the slower ~2-min LLM ticks so fast moves are cut promptly.
        if not is_market_open(now_utc()):
            return
        try:
            self._day_agent().manage_positions_only()
        except Exception:
            log.exception("stop_monitor failed")

    def job_scan_refresh(self) -> None:
        # Refresh the intraday scanner cache so list_ideas returns FRESH ideas
        # (and prices) instead of the last scan. Without this the idea universe
        # freezes at whenever scan_run last ran, and the agent keeps seeing stale
        # names/prices. Market-hours only. The upstream server keeps working even
        # if our client wait cap elapses, so a timeout here is benign.
        if not is_market_open(now_utc()):
            return
        st = self.short_term
        if st is None or not hasattr(st, "scan_run"):
            return
        timeout = float(getattr(self.settings, "scan_refresh_timeout_seconds", 300.0) or 300.0)
        universe = str(getattr(self.settings, "scan_universe", "liquid") or "liquid")
        t0 = time.monotonic()
        try:
            st.scan_run(mode="intraday", universe=universe, timeout=timeout)
            log.info("scan_refresh: %s ideas refreshed in %.0fs",
                      universe, time.monotonic() - t0)
        except TimeoutError:
            log.info("scan_refresh: client wait cap hit after %.0fs (server keeps "
                      "scanning; cache updates when done)", time.monotonic() - t0)
        except Exception:
            log.exception("scan_refresh failed")

    def job_long_tick(self) -> None:
        try:
            self._long_agent().run_once()
        except Exception:
            log.exception("long_tick failed")

    def job_coord_tick(self) -> None:
        # Gate on "first trading week of the month".
        if now_utc().day > 7:
            return
        try:
            self._coordinator().run_once()
        except Exception:
            log.exception("coord_tick failed")

    def job_tick_poll(self) -> None:
        # Heartbeat: record that the scheduler is alive so the UI can report it
        # (this poll runs every 5s). Best-effort; never let it break the poll.
        try:
            dbm.set_setting(self.conn, "scheduler_heartbeat", now_utc().isoformat())
        except Exception:
            log.debug("heartbeat write failed", exc_info=True)
        try:
            rows = self.conn.execute(
                "SELECT id, agent FROM tick_requests WHERE consumed_at IS NULL "
                "ORDER BY id LIMIT 50"
            ).fetchall()
        except Exception:
            log.exception("tick_poll select failed")
            return
        if not rows:
            return
        # Coalesce: multiple pending requests for the SAME agent (e.g. the user
        # clicking "Tick now" several times) only need ONE run. Run each distinct
        # agent at most once and mark ALL of that agent's pending requests
        # consumed — so repeated clicks are cheap and resolve in one cycle instead
        # of queuing N sequential LLM ticks.
        by_agent: dict[str, list[int]] = {}
        for row in rows:
            by_agent.setdefault(row["agent"], []).append(row["id"])
        now_iso = now_utc().isoformat()
        for agent, ids in by_agent.items():
            try:
                if agent == "day":
                    self._day_agent().run_once()
                elif agent == "long":
                    self._long_agent().run_once()
                elif agent == "coordinator":
                    self._coordinator().run_once()
                elif agent == "mtm":
                    self.job_mtm()
                else:
                    log.warning("tick_poll: unknown agent=%r", agent)
            except Exception:
                log.exception("tick (agent=%s) failed", agent)
            finally:
                placeholders = ",".join("?" * len(ids))
                self.conn.execute(
                    f"UPDATE tick_requests SET consumed_at=? WHERE id IN ({placeholders})",
                    (now_iso, *ids),
                )

    # ---------------- wiring ----------------

    def register_jobs(self) -> None:
        self.scheduler.add_job(self.job_mtm, IntervalTrigger(minutes=1), id="mtm")
        self.scheduler.add_job(self.job_reconcile, IntervalTrigger(minutes=1), id="reconcile")
        # Day-tick cadence is configurable (DAY_TICK_SECONDS, default 60s) so the
        # book can be traded at higher frequency when the LLM quota allows. Values
        # below 60s only help if the model provider isn't rate-limiting; the
        # provider retries/downshifts and the UI shows a throttle notice otherwise.
        day_secs = max(5, int(getattr(self.settings, "day_tick_seconds", 60) or 60))
        self.scheduler.add_job(self.job_day_tick, IntervalTrigger(seconds=day_secs),
                                 id="day_tick", max_instances=1, coalesce=True)
        log.info("day_tick cadence: every %ds", day_secs)
        # Keep the intraday idea universe fresh (list_ideas only returns the last
        # scan_run). Runs on its own thread so a slow scan doesn't wedge day ticks.
        scan_secs = int(getattr(self.settings, "scan_refresh_seconds", 300) or 0)
        if scan_secs > 0:
            self.scheduler.add_job(self.job_scan_refresh,
                                     IntervalTrigger(seconds=max(60, scan_secs)),
                                     id="scan_refresh", max_instances=1, coalesce=True,
                                     next_run_time=now_utc())
            log.info("scan_refresh cadence: every %ds", max(60, scan_secs))
        # Fast, LLM-free stop monitor between the slower day ticks.
        stop_secs = int(getattr(self.settings, "stop_monitor_seconds", 30) or 0)
        if stop_secs > 0:
            self.scheduler.add_job(self.job_stop_monitor,
                                     IntervalTrigger(seconds=max(10, stop_secs)),
                                     id="stop_monitor", max_instances=1, coalesce=True)
            log.info("stop_monitor cadence: every %ds", max(10, stop_secs))
        self.scheduler.add_job(self.job_long_tick,
                                 CronTrigger(hour=21, minute=30), id="long_tick")
        self.scheduler.add_job(self.job_coord_tick,
                                 CronTrigger(hour=21, minute=45), id="coord_tick")
        self.scheduler.add_job(self.job_tick_poll,
                                 IntervalTrigger(seconds=5), id="tick_poll")

    def _start_day_tick_watchdog(self) -> None:
        """Self-healing guard against a hung day tick.

        ``max_instances=1`` means a single day tick that hangs on an unbounded
        network/MCP call blocks *every* future tick — the silent multi-hour stall
        seen before. The internal LLM + placement deadlines bound the common
        cases, but a truly stuck I/O call would still wedge the job slot. This
        watchdog exits the process past a hard wall-clock bound so Docker
        (``restart: unless-stopped``) brings up a fresh scheduler and trading
        resumes — the same recovery as a manual restart, but automatic.
        """
        day_secs = max(5, int(getattr(self.settings, "day_tick_seconds", 60) or 60))
        limit = max(240.0, day_secs * 3.0)

        def _watch() -> None:
            while not self._stop.wait(15.0):
                started = self._day_tick_started_at
                if started is None:
                    continue
                elapsed = time.monotonic() - started
                if elapsed > limit:
                    log.error("day_tick hung for %.0fs (limit %.0fs); exiting for "
                               "a clean restart so trading resumes", elapsed, limit)
                    os._exit(1)

        t = threading.Thread(target=_watch, name="day-tick-watchdog", daemon=True)
        t.start()
        log.info("day-tick watchdog armed (hard limit %.0fs)", limit)

    def install_signals(self) -> None:
        def handler(signum, _frame):
            log.info("signal %s received; shutting down", signum)
            try:
                self.scheduler.shutdown(wait=False)
            except Exception:
                pass
            self._stop.set()
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def run(self) -> None:
        self._acquire_lock()
        try:
            self.setup()
            self.register_jobs()
            self.install_signals()
            self._start_day_tick_watchdog()
            log.info("scheduler starting (broker=%s)", self.settings.broker_backend)
            self.scheduler.start()
        finally:
            self.teardown()

    def start_background(self) -> None:
        """Start a non-blocking BackgroundScheduler (for Streamlit Cloud).

        Skips the pid-file lock since the host process is Streamlit itself.
        """
        if not self.background:
            raise RuntimeError("start_background requires background=True")
        self.setup()
        self.register_jobs()
        self.scheduler.start()
        log.info("background scheduler running (broker=%s)", self.settings.broker_backend)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(name)s %(message)s")
    SchedulerRunner().run()


if __name__ == "__main__":
    main()
