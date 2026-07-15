"""Alpaca paper broker — delegates to the upstream stock-recommender MCP.

This impl never imports an Alpaca SDK; it routes every write through the
sibling repo's MCP server, which is itself hard-coded to paper. We mirror
each call into our local `orders` table with `venue='alpaca_paper'` and
`status='routed_external'` (then update to the resolved status once the
MCP returns).

For Phase 2 we accept a `MCPClient`-shaped object via DI so the unit tests
can use a stub. The real client is wired in Phase 3.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Protocol

import pandas as pd

from src.brokers.base import (
    AccountSnapshot,
    BrokerBase,
    OrderRequest,
    OrderResult,
    Position,
)
from src.config import get_settings
from src.sandbox import db as dbm

log = logging.getLogger(__name__)

_SUB_TO_ACCOUNT = {"day": "day_alpaca", "long": "long_alpaca"}


class LongTermMCPLike(Protocol):
    def get_account(self) -> dict: ...
    def list_positions(self) -> list[dict]: ...
    def place_order(self, *, symbol: str, qty: float, side: str,
                    order_type: str = "market",
                    time_in_force: str = "day",
                    limit_price: float | None = None) -> dict: ...
    def cancel_order(self, order_id: str) -> dict: ...
    def close_position(self, symbol: str, percentage: int = 100) -> dict: ...


class AlpacaPaperBroker(BrokerBase):
    name = "alpaca_paper"

    def __init__(self, mcp: LongTermMCPLike, conn: sqlite3.Connection | None = None) -> None:
        if conn is None:
            conn = dbm.get_conn()
            dbm.migrate(conn)
            dbm.bootstrap_accounts(conn)
        self.conn = conn
        self.mcp = mcp

    # ---------- reads ----------

    def get_account(self, sub_account: str = "day") -> AccountSnapshot:
        # Alpaca has one account; we present it under both day_alpaca/long_alpaca names.
        acct = self.mcp.get_account()
        equity = float(acct.get("equity", 0.0))
        cash = float(acct.get("cash", 0.0))
        return AccountSnapshot(
            name=_SUB_TO_ACCOUNT.get(sub_account, sub_account),
            venue=self.name, equity=equity, cash=cash, positions_value=equity - cash,
        )

    def list_positions(self, sub_account: str = "day") -> list[Position]:
        out: list[Position] = []
        for p in self.mcp.list_positions():
            qty = float(p.get("qty", 0.0))
            avg = float(p.get("entry_price", p.get("avg_entry_price", 0.0)))
            mark = float(p.get("current_price", p.get("market_price", avg)))
            out.append(Position(
                symbol=str(p.get("symbol", "")).upper(),
                qty=qty, avg_cost=avg, mark_price=mark,
                unrealized_pnl=(mark - avg) * qty,
            ))
        return out

    # ---------- writes ----------

    @staticmethod
    def _is_trading_disabled(resp: object) -> bool:
        return isinstance(resp, dict) and str(resp.get("blocked") or "") == "trading_disabled"

    @staticmethod
    def _resp_reason(resp: object) -> str:
        if not isinstance(resp, dict):
            return ""
        return str(resp.get("blocked") or resp.get("error") or "").lower()

    # Reject reasons that a retry cannot fix — the venue will refuse them every
    # time (bad symbol, not shortable, insufficient funds, cap breaches). We do
    # NOT retry these; everything else transient (trading_disabled, transport
    # errors, empty responses) is retried with backoff.
    _PERMANENT_MARKERS = (
        "cannot be sold short", "not shortable", "insufficient", "buying power",
        "sandbox_violation", "cap", "forbidden", "invalid", "unprocessable",
        "not tradable", "not_tradable", "asset", "422",
    )

    def _is_permanent_reject(self, resp: object) -> bool:
        reason = self._resp_reason(resp)
        return any(m in reason for m in self._PERMANENT_MARKERS)

    def _place_via_mcp(self, req: OrderRequest) -> dict:
        """Submit through the upstream MCP, retrying TRANSIENT failures.

        Alpaca is the more important leg, so we work hard to land the order:
          - `trading_disabled` (a spurious block from a degraded long-lived MCP
            client) → restart the MCP client and retry.
          - transport exceptions / empty or id-less responses → retry with
            exponential backoff.
          - a genuinely permanent reject (not shortable, insufficient funds, cap,
            bad symbol) → return immediately; retrying can't help.
        """
        s = get_settings()
        max_attempts = max(1, int(getattr(s, "alpaca_max_retries", 4) or 1))
        backoff = float(getattr(s, "alpaca_retry_backoff", 1.0) or 1.0)
        cap = float(getattr(s, "alpaca_retry_cap", 6.0) or 6.0)
        # Overall wall-clock budget for the whole retry sequence: a slow MCP
        # restart + retries must NOT block the scheduler tick for minutes (that
        # recreated the multi-minute stall). Give up cleanly past the budget.
        time_budget = float(getattr(s, "alpaca_place_budget_seconds", 25.0) or 25.0)

        def _call() -> dict:
            return self.mcp.place_order(
                symbol=req.symbol.upper(), qty=req.qty, side=req.side,
                order_type=req.order_type, time_in_force=req.tif,
                limit_price=req.limit_price,
            )

        started = time.monotonic()
        restarted = False  # restart the MCP client at MOST once per order
        last_resp: dict = {}
        for attempt in range(max_attempts):
            try:
                resp = _call()
            except Exception:
                log.warning("alpaca place_order transport error (attempt %d/%d) for %s",
                             attempt + 1, max_attempts, req.symbol, exc_info=True)
                resp = {}
            last_resp = resp if isinstance(resp, dict) else {}

            ext_id = (str(resp.get("order_id") or resp.get("id") or "")
                      if isinstance(resp, dict) else "")
            if ext_id and not self._is_trading_disabled(resp):
                return resp  # success — order accepted by Alpaca

            if self._is_permanent_reject(resp):
                return resp  # venue will refuse every time; don't waste retries

            # Out of time budget → stop retrying so the tick isn't blocked.
            if time.monotonic() - started > time_budget:
                log.warning("alpaca place_order budget (%.0fs) exhausted for %s; giving up",
                             time_budget, req.symbol)
                break

            # Transient. If it's the spurious trading gate, restart the MCP client
            # ONCE (a fresh client reliably clears it) before retrying.
            if self._is_trading_disabled(resp) and not restarted:
                restarted = True
                log.warning("alpaca leg trading_disabled (attempt %d/%d); "
                             "restarting MCP client", attempt + 1, max_attempts)
                try:
                    self.mcp.stop()
                except Exception:
                    log.debug("mcp.stop() during self-heal failed", exc_info=True)
                try:
                    self.mcp.start()
                except Exception:
                    log.exception("mcp.start() during self-heal failed")
            if attempt < max_attempts - 1:
                time.sleep(min(backoff * (2 ** attempt), cap))
        log.error("alpaca place_order failed after %d attempts for %s: %s",
                   max_attempts, req.symbol, str(last_resp)[:160])
        return last_resp

    def place_order(self, req: OrderRequest) -> OrderResult:
        aid = dbm.get_account_id(self.conn, _SUB_TO_ACCOUNT.get(req.sub_account, req.sub_account))
        now_s = datetime.now(timezone.utc).isoformat()

        # Record routed_external first so a crash still leaves an audit row.
        cur = self.conn.execute(
            """
            INSERT INTO orders(account_id, ts, symbol, side, qty, order_type, limit_price, tif,
                               status, submitted_at, agent, thesis, venue, dual_group_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'routed_external', ?, ?, ?, ?, ?)
            """,
            (aid, now_s, req.symbol.upper(), req.side, req.qty, req.order_type,
             req.limit_price, req.tif, now_s, req.agent, req.thesis,
             self.name, req.dual_group_id),
        )
        oid = cur.lastrowid

        try:
            resp = self._place_via_mcp(req)
        except Exception as exc:
            log.exception("alpaca place_order failed for order %s", oid)
            self.conn.execute("UPDATE orders SET status='rejected' WHERE id=?", (oid,))
            return OrderResult(id=oid, external_id=None, status="rejected",
                               fill_price=None, fees=0.0, venue=self.name,
                               dual_group_id=req.dual_group_id)

        # Detect a non-order response: the upstream returns a blocked/dry-run/error
        # dict (no order id/status) when trading is disabled or a cap is hit. Treat
        # it as a failure instead of silently leaving the row as 'routed_external'.
        ext_id = str(resp.get("order_id") or resp.get("id") or "") if isinstance(resp, dict) else ""
        blocked = isinstance(resp, dict) and (
            "blocked" in resp or "error" in resp or resp.get("dry_run") is True
        )
        if blocked or not ext_id:
            reason = "unknown"
            if isinstance(resp, dict):
                reason = str(resp.get("blocked") or resp.get("error")
                             or ("dry_run" if resp.get("dry_run") else "no_order_id"))
            note = ""
            if isinstance(resp, dict):
                note = str(resp.get("message") or resp.get("blocked") or resp.get("error"))[:200]
            log.error("alpaca order %s NOT placed (%s): %s", oid, reason, note)
            self.conn.execute(
                "UPDATE orders SET status='rejected', thesis=? WHERE id=?",
                (f"alpaca_not_placed:{reason}|{req.thesis or ''}"[:500], oid),
            )
            return OrderResult(id=oid, external_id=None, status="rejected",
                               fill_price=None, fees=0.0, venue=self.name,
                               dual_group_id=req.dual_group_id)

        status = str(resp.get("status", "accepted"))
        fill_price = resp.get("filled_avg_price") or resp.get("fill_price")
        fill_price = float(fill_price) if fill_price is not None else None
        filled_at = resp.get("filled_at") or (now_s if status == "filled" else None)

        self.conn.execute(
            "UPDATE orders SET external_id=?, status=?, fill_price=?, filled_at=? WHERE id=?",
            (ext_id or None, status, fill_price, filled_at, oid),
        )

        return OrderResult(id=oid, external_id=ext_id or None, status=status,
                           fill_price=fill_price, fees=0.0, venue=self.name,
                           dual_group_id=req.dual_group_id)

    def cancel_order(self, order_id: int) -> None:
        row = self.conn.execute(
            "SELECT external_id FROM orders WHERE id=? AND venue=?",
            (order_id, self.name),
        ).fetchone()
        if row and row["external_id"]:
            try:
                self.mcp.cancel_order(row["external_id"])
            except Exception:
                log.exception("alpaca cancel_order failed for %s", order_id)
        self.conn.execute(
            "UPDATE orders SET status='cancelled' WHERE id=? AND status IN ('pending','routed_external')",
            (order_id,),
        )

    def close_position(self, symbol: str, sub_account: str = "day",
                       percentage: float = 100.0) -> OrderResult:
        aid = dbm.get_account_id(self.conn, _SUB_TO_ACCOUNT.get(sub_account, sub_account))
        now_s = datetime.now(timezone.utc).isoformat()
        cur = self.conn.execute(
            """
            INSERT INTO orders(account_id, ts, symbol, side, qty, order_type, tif,
                               status, submitted_at, agent, thesis, venue)
            VALUES (?, ?, ?, 'sell', 0, 'market', 'day', 'routed_external', ?, 'manual', 'close_position', ?)
            """,
            (aid, now_s, symbol.upper(), now_s, self.name),
        )
        oid = cur.lastrowid
        resp = self._close_via_mcp(symbol.upper(), percentage)
        ext_id = str(resp.get("order_id") or resp.get("id") or "") if isinstance(resp, dict) else ""
        blocked = isinstance(resp, dict) and ("blocked" in resp or "error" in resp)
        if blocked or not ext_id:
            reason = "unknown"
            if isinstance(resp, dict):
                reason = str(resp.get("blocked") or resp.get("error") or "no_order_id")
            log.error("alpaca close_position %s NOT placed (%s)", oid, reason)
            self.conn.execute("UPDATE orders SET status='rejected' WHERE id=?", (oid,))
            return OrderResult(id=oid, external_id=None, status="rejected",
                               fill_price=None, fees=0.0, venue=self.name)
        status = str(resp.get("status", "accepted"))
        self.conn.execute(
            "UPDATE orders SET external_id=?, status=? WHERE id=?",
            (ext_id or None, status, oid),
        )
        return OrderResult(id=oid, external_id=ext_id or None, status=status,
                           fill_price=None, fees=0.0, venue=self.name)

    def _close_via_mcp(self, symbol: str, percentage: float) -> dict:
        """Close a position through the upstream MCP, retrying TRANSIENT failures.

        Closes are risk-reducing (stop-losses, exits, force-flat), so a spurious
        `trading_disabled` or transport error must NOT be allowed to abandon the
        close and leave the position open. Mirrors `_place_via_mcp`: restart the
        MCP client once on `trading_disabled`, retry transient errors with
        backoff, bail on a permanent reject (e.g. position already flat), and
        respect an overall wall-clock budget so a tick isn't blocked for minutes.
        """
        s = get_settings()
        max_attempts = max(1, int(getattr(s, "alpaca_max_retries", 4) or 1))
        backoff = float(getattr(s, "alpaca_retry_backoff", 1.0) or 1.0)
        cap = float(getattr(s, "alpaca_retry_cap", 6.0) or 6.0)
        time_budget = float(getattr(s, "alpaca_place_budget_seconds", 25.0) or 25.0)

        started = time.monotonic()
        restarted = False
        last_resp: dict = {}
        for attempt in range(max_attempts):
            try:
                resp = self.mcp.close_position(symbol, percentage=int(percentage))
            except Exception:
                log.warning("alpaca close_position transport error (attempt %d/%d) for %s",
                             attempt + 1, max_attempts, symbol, exc_info=True)
                resp = {}
            last_resp = resp if isinstance(resp, dict) else {}

            ext_id = (str(resp.get("order_id") or resp.get("id") or "")
                      if isinstance(resp, dict) else "")
            if ext_id and not self._is_trading_disabled(resp):
                return resp  # close accepted by Alpaca

            if self._is_permanent_reject(resp):
                return resp  # e.g. position already flat / not closable — retry won't help

            if time.monotonic() - started > time_budget:
                log.warning("alpaca close_position budget (%.0fs) exhausted for %s; giving up",
                             time_budget, symbol)
                break

            if self._is_trading_disabled(resp) and not restarted:
                restarted = True
                log.warning("alpaca close_position trading_disabled (attempt %d/%d); "
                             "restarting MCP client", attempt + 1, max_attempts)
                try:
                    self.mcp.stop()
                except Exception:
                    log.debug("mcp.stop() during close self-heal failed", exc_info=True)
                try:
                    self.mcp.start()
                except Exception:
                    log.exception("mcp.start() during close self-heal failed")
            if attempt < max_attempts - 1:
                time.sleep(min(backoff * (2 ** attempt), cap))
        log.error("alpaca close_position failed after %d attempts for %s: %s",
                   max_attempts, symbol, str(last_resp)[:160])
        return last_resp


    def mark_to_market(self, now: datetime, sub_account: str = "day") -> AccountSnapshot:
        aid = dbm.get_account_id(self.conn, _SUB_TO_ACCOUNT.get(sub_account, sub_account))
        snap = self.get_account(sub_account)
        now_s = now.isoformat()
        # Persist positions snapshots for the UI / divergence analysis.
        for p in self.list_positions(sub_account):
            self.conn.execute(
                "INSERT INTO positions_snapshot(account_id, ts, symbol, qty, avg_cost, mark_price) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (aid, now_s, p.symbol, p.qty, p.avg_cost, p.mark_price),
            )
        self.conn.execute(
            "INSERT INTO equity_curve(account_id, ts, cash, positions_value, equity) VALUES (?, ?, ?, ?, ?)",
            (aid, now_s, snap.cash, snap.positions_value, snap.equity),
        )
        return snap

    def reconcile(self) -> int:
        """Pull recent Alpaca order statuses and sync local `alpaca_paper` rows.

        Orders submitted as market/limit may fill asynchronously; the local row
        is stamped with the status Alpaca returned at submit time (often
        'accepted'/'new'). This polls Alpaca for the resolved status and updates
        the mirror so the UI reflects real fills. Returns rows updated.
        """
        try:
            orders = self.mcp.list_orders(status="all", limit=100)
        except Exception:
            log.exception("alpaca reconcile: list_orders failed")
            return 0
        by_ext: dict[str, dict] = {}
        for o in orders or []:
            oid = str(o.get("id") or o.get("order_id") or "")
            if oid:
                by_ext[oid] = o
        if not by_ext:
            return 0
        rows = self.conn.execute(
            "SELECT id, external_id, status FROM orders WHERE venue=? "
            "AND external_id IS NOT NULL "
            "AND status NOT IN ('filled','cancelled','rejected','expired','canceled')",
            (self.name,),
        ).fetchall()
        updated = 0
        for r in rows:
            o = by_ext.get(str(r["external_id"]))
            if not o:
                continue
            status = str(o.get("status") or "")
            if not status or status == r["status"]:
                continue
            fp = o.get("filled_avg_price") or o.get("fill_price")
            try:
                fp = float(fp) if fp not in (None, "") else None
            except (TypeError, ValueError):
                fp = None
            filled_at = o.get("filled_at")
            self.conn.execute(
                "UPDATE orders SET status=?, "
                "fill_price=COALESCE(?, fill_price), "
                "filled_at=COALESCE(?, filled_at) WHERE id=?",
                (status, fp, filled_at, r["id"]),
            )
            updated += 1
        if updated:
            log.info("alpaca reconcile: updated %d order row(s)", updated)

        # Age out orders that were routed but never got an external_id (e.g. a
        # close_position that returned no order id). Without an id they can never
        # resolve, so after 2 minutes mark them failed rather than leaving them
        # 'routed_external' forever and skewing the "unresolved" view.
        stale = self.conn.execute(
            "UPDATE orders SET status='expired', thesis=COALESCE(thesis,'')||'|reconcile:no_external_id' "
            "WHERE venue=? AND external_id IS NULL "
            "AND status IN ('routed_external','pending_new','new','accepted') "
            "AND submitted_at < datetime('now','-2 minutes')",
            (self.name,),
        ).rowcount
        if stale:
            log.info("alpaca reconcile: aged out %d stale id-less order(s)", stale)
            updated += stale
        return updated

    def equity_curve(self, sub_account: str = "day",
                     since: datetime | None = None) -> pd.DataFrame:
        aid = dbm.get_account_id(self.conn, _SUB_TO_ACCOUNT.get(sub_account, sub_account))
        if since:
            df = pd.read_sql_query(
                "SELECT ts, cash, positions_value, equity FROM equity_curve "
                "WHERE account_id=? AND ts >= ? ORDER BY ts",
                self.conn, params=(aid, since.isoformat()),
            )
        else:
            df = pd.read_sql_query(
                "SELECT ts, cash, positions_value, equity FROM equity_curve "
                "WHERE account_id=? ORDER BY ts",
                self.conn, params=(aid,),
            )
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"])
        return df
