"""Dual broker — fan every write out to both `primary` and `secondary` legs.

Contract:
- `place_order(req)` mints a `dual_group_id` (UUID4) and submits to both
  legs sequentially (primary then secondary). Both legs write their own
  `orders` row stamped with that group id. Sequential rather than parallel
  because SQLite connections are not safe for concurrent access from
  multiple threads on the same `Connection` object; the per-trade latency
  cost is negligible (<1 s typical).
- If the **primary** raises, the call raises (the agent must see the failure).
  If only the **secondary** raises, the primary order stands and a
  `dual_divergence` row is recorded — never roll back the primary.
- `cancel_order` / `close_position` look up the `dual_group_id` for the
  given order row and call cancel/close on both legs.
- Read methods default to the primary; pass `venue='secondary'` for the
  Alpaca-mirror view.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd

from src.brokers.base import (
    AccountSnapshot,
    BrokerBase,
    OrderRequest,
    OrderResult,
    Position,
)
from src.sandbox import db as dbm

log = logging.getLogger(__name__)


class DualBroker(BrokerBase):
    name = "dual"

    def __init__(self, primary: BrokerBase, secondary: BrokerBase,
                 conn: sqlite3.Connection | None = None) -> None:
        self.primary = primary
        self.secondary = secondary
        self.conn = conn or getattr(primary, "conn", None) or dbm.get_conn()
        if self.conn is None:
            self.conn = dbm.get_conn()
            dbm.migrate(self.conn)
            dbm.bootstrap_accounts(self.conn)

    # ---------- reads (default to primary) ----------

    def get_account(self, sub_account: str = "day", venue: str = "primary") -> AccountSnapshot:
        return (self.primary if venue == "primary" else self.secondary).get_account(sub_account)

    def list_positions(self, sub_account: str = "day", venue: str = "primary") -> list[Position]:
        return (self.primary if venue == "primary" else self.secondary).list_positions(sub_account)

    def equity_curve(self, sub_account: str = "day", since=None, venue: str = "primary") -> pd.DataFrame:
        return (self.primary if venue == "primary" else self.secondary).equity_curve(sub_account, since)

    # ---------- writes (fan out) ----------

    def place_order(self, req: OrderRequest) -> OrderResult:
        group_id = req.dual_group_id or uuid.uuid4().hex
        prim_req = replace(req, dual_group_id=group_id)
        sec_req = replace(req, dual_group_id=group_id)

        # Primary first — its exception must propagate.
        prim_result = self.primary.place_order(prim_req)

        try:
            sec_result = self.secondary.place_order(sec_req)
        except Exception as exc:
            log.exception("secondary broker raised in dual place_order")
            self._record_divergence(group_id, "secondary_error",
                                    primary_val=prim_result.status,
                                    secondary_val=type(exc).__name__,
                                    note=str(exc)[:200])
            return prim_result

        self._maybe_record_fill_divergence(group_id, prim_result, sec_result)
        return prim_result

    def cancel_order(self, order_id: int) -> None:
        row = self.conn.execute(
            "SELECT dual_group_id FROM orders WHERE id = ?", (order_id,),
        ).fetchone()
        if not row or not row["dual_group_id"]:
            v = self.conn.execute("SELECT venue FROM orders WHERE id = ?", (order_id,)).fetchone()
            if v and v["venue"] == "alpaca_paper":
                self.secondary.cancel_order(order_id)
            else:
                self.primary.cancel_order(order_id)
            return

        group = row["dual_group_id"]
        for venue, broker in (("sandbox", self.primary), ("alpaca_paper", self.secondary)):
            r = self.conn.execute(
                "SELECT id FROM orders WHERE dual_group_id = ? AND venue = ?",
                (group, venue),
            ).fetchone()
            if r:
                try:
                    broker.cancel_order(int(r["id"]))
                except Exception:
                    log.exception("dual cancel: %s leg failed", venue)
                    self._record_divergence(group, "secondary_error",
                                            primary_val="cancel", secondary_val="error",
                                            note=f"{venue} cancel failed")

    def close_position(self, symbol: str, sub_account: str = "day",
                       percentage: float = 100.0) -> OrderResult:
        group_id = uuid.uuid4().hex
        prim = self._close_with_group(self.primary, symbol, sub_account, percentage, group_id)
        try:
            sec = self._close_with_group(self.secondary, symbol, sub_account, percentage, group_id)
        except Exception as exc:
            self._record_divergence(group_id, "secondary_error",
                                    primary_val=prim.status, secondary_val=type(exc).__name__,
                                    note=str(exc)[:200])
            return prim
        self._maybe_record_fill_divergence(group_id, prim, sec)
        return prim

    def _close_with_group(self, broker: BrokerBase, symbol: str, sub_account: str,
                          percentage: float, group_id: str) -> OrderResult:
        result = broker.close_position(symbol, sub_account, percentage)
        self.conn.execute(
            "UPDATE orders SET dual_group_id=? WHERE id=?",
            (group_id, result.id),
        )
        return replace(result, dual_group_id=group_id)

    # ---------- MTM (both legs) ----------

    def mark_to_market(self, now: datetime, sub_account: str = "day") -> AccountSnapshot:
        prim = self.primary.mark_to_market(now, sub_account)
        try:
            self.secondary.mark_to_market(now, sub_account)
        except Exception:
            log.exception("secondary mark_to_market failed")
        return prim

    # ---------- divergence helpers ----------

    def _record_divergence(self, group_id: str, kind: str, *, primary_val: str | None,
                            secondary_val: str | None, note: str = "") -> None:
        self.conn.execute(
            "INSERT INTO dual_divergence(dual_group_id, ts, kind, primary_val, secondary_val, note) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (group_id, datetime.now(timezone.utc).isoformat(), kind,
             str(primary_val) if primary_val is not None else None,
             str(secondary_val) if secondary_val is not None else None,
             note),
        )

    def _maybe_record_fill_divergence(self, group_id: str, prim: OrderResult,
                                       sec: OrderResult) -> None:
        if prim.status != sec.status:
            self._record_divergence(group_id, "status",
                                    primary_val=prim.status, secondary_val=sec.status)
        if (prim.fill_price is not None and sec.fill_price is not None
                and abs(prim.fill_price - sec.fill_price) > 0.005 * prim.fill_price):
            self._record_divergence(group_id, "fill_price",
                                    primary_val=f"{prim.fill_price:.4f}",
                                    secondary_val=f"{sec.fill_price:.4f}",
                                    note="diff > 50 bps")
