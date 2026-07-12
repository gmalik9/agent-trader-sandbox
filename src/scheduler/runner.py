"""Scheduler runner — long-running process that owns all writes.

Jobs (UTC scheduling; market-hour gating is internal to each job):
- `mtm`        every 1 min — mark-to-market both sub-accounts
- `day_tick`   every 5 min — DayTraderAgent.run_once
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
from src.signals.local import LocalLongTermClient, LocalShortTermClient

log = logging.getLogger(__name__)
LOCK_PATH = DATA_DIR / "scheduler.lock"


class SchedulerRunner:
    def __init__(self, *, background: bool = False) -> None:
        self.settings = get_settings()
        self.conn: sqlite3.Connection | None = None
        self.short_term: ShortTermClient | LocalShortTermClient | None = None
        self.long_term: LongTermClient | LocalLongTermClient | None = None
        self.broker = None
        self.provider = None
        self.background = background
        cls = BackgroundScheduler if background else BlockingScheduler
        self.scheduler = cls(
            timezone="UTC", job_defaults={"coalesce": True, "max_instances": 1,
                                            "misfire_grace_time": 60},
        )
        self._stop = threading.Event()

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

        try:
            self.provider = get_provider()
            log.info("LLM provider: %s / %s", self.provider.name, self.provider.model)
        except Exception as e:
            log.warning("no LLM provider configured (%s); LLM-driven ticks will error", e)
            self.provider = None

    def _make_short_term(self) -> ShortTermClient | LocalShortTermClient:
        """Real short-term MCP client if configured & healthy, else local fallback."""
        if self.settings.short_term_trader_path:
            try:
                client = ShortTermClient()
                client.start()
                log.info("short-term MCP client up")
                return client
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
                                provider=self.provider)

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

    def job_day_tick(self) -> None:
        if not is_market_open(now_utc()):
            return
        try:
            self._day_agent().run_once()
        except Exception:
            log.exception("day_tick failed")

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
        try:
            rows = self.conn.execute(
                "SELECT id, agent FROM tick_requests WHERE consumed_at IS NULL "
                "ORDER BY id LIMIT 20"
            ).fetchall()
        except Exception:
            log.exception("tick_poll select failed")
            return
        for row in rows:
            tid, agent = row["id"], row["agent"]
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
                log.exception("tick %s (agent=%s) failed", tid, agent)
            finally:
                self.conn.execute(
                    "UPDATE tick_requests SET consumed_at=? WHERE id=?",
                    (now_utc().isoformat(), tid),
                )

    # ---------------- wiring ----------------

    def register_jobs(self) -> None:
        self.scheduler.add_job(self.job_mtm, IntervalTrigger(minutes=1), id="mtm")
        self.scheduler.add_job(self.job_day_tick, IntervalTrigger(minutes=5), id="day_tick")
        self.scheduler.add_job(self.job_long_tick,
                                 CronTrigger(hour=21, minute=30), id="long_tick")
        self.scheduler.add_job(self.job_coord_tick,
                                 CronTrigger(hour=21, minute=45), id="coord_tick")
        self.scheduler.add_job(self.job_tick_poll,
                                 IntervalTrigger(seconds=5), id="tick_poll")

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
