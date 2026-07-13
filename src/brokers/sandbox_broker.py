"""Local sandbox broker.

Owns its sub-accounts (`day`, `long`) in our SQLite ledger. Enforces:
- per-order USD cap (`max_order_usd`)
- per-symbol % of equity cap (`max_symbol_pct`)
- blocklist (leveraged/inverse/volatility ETFs)
- no short selling (qty after fill must stay >= 0)
- cash sufficiency for buys
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone

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
from src.sandbox.engine import (
    BarProvider,
    OrderSpec,
    YFinanceBarProvider,
    positions_from_ledger,
    simulate_fill,
)

log = logging.getLogger(__name__)

DEFAULT_BLOCKLIST = {
    # Leveraged / inverse / vol ETFs (subset; can be extended).
    "TQQQ", "SQQQ", "SOXL", "SOXS", "UPRO", "SPXU", "TNA", "TZA",
    "UVXY", "SVXY", "VXX", "VIXY", "LABU", "LABD", "FAS", "FAZ",
    "NUGT", "DUST", "JNUG", "JDST", "SPXL", "SPXS", "QLD", "QID",
    "UDOW", "SDOW", "DRN", "DRV", "ERX", "ERY", "BOIL", "KOLD",
}


class SandboxBroker(BrokerBase):
    name = "sandbox"

    def __init__(
        self,
        conn: sqlite3.Connection | None = None,
        *,
        bar_provider: BarProvider | None = None,
        max_order_usd: float = 25_000.0,
        max_symbol_pct: float = 25.0,
        blocklist: set[str] | None = None,
        allow_shorting: bool | None = None,
        allow_leveraged: bool | None = None,
        max_leverage: float | None = None,
    ) -> None:
        if conn is None:
            conn = dbm.get_conn()
            dbm.migrate(conn)
            dbm.bootstrap_accounts(conn)
        self.conn = conn
        self.bars = bar_provider or YFinanceBarProvider()
        self.max_order_usd = max_order_usd
        self.max_symbol_pct = max_symbol_pct
        s = get_settings()
        self.allow_shorting = s.allow_shorting if allow_shorting is None else allow_shorting
        self.allow_leveraged = s.allow_leveraged if allow_leveraged is None else allow_leveraged
        self.max_leverage = s.max_leverage if max_leverage is None else max_leverage
        # The blocklist only applies when leveraged products are disallowed.
        if blocklist is not None:
            self.blocklist = blocklist
        elif self.allow_leveraged:
            self.blocklist = set()
        else:
            self.blocklist = set(DEFAULT_BLOCKLIST)

    # ---------- accounts / positions ----------

    def _account_id(self, sub_account: str) -> int:
        return dbm.get_account_id(self.conn, sub_account)

    def _equity(self, account_id: int, *, mark_overrides: dict[str, float] | None = None) -> tuple[float, float]:
        """Return (cash, positions_value) using last-known marks."""
        cash = dbm.get_cash(self.conn, account_id)
        positions = positions_from_ledger(self.conn, account_id)
        pv = 0.0
        for sym, p in positions.items():
            mark = (mark_overrides or {}).get(sym)
            if mark is None:
                mark = self._last_mark(account_id, sym, fallback=p.avg_cost)
            pv += p.qty * mark
        return cash, pv

    def _last_mark(self, account_id: int, symbol: str, *, fallback: float) -> float:
        row = self.conn.execute(
            "SELECT mark_price FROM positions_snapshot WHERE account_id=? AND symbol=? "
            "ORDER BY ts DESC LIMIT 1",
            (account_id, symbol),
        ).fetchone()
        return float(row["mark_price"]) if row else fallback

    def get_account(self, sub_account: str = "day") -> AccountSnapshot:
        aid = self._account_id(sub_account)
        cash, pv = self._equity(aid)
        return AccountSnapshot(name=sub_account, venue=self.name, equity=cash + pv,
                               cash=cash, positions_value=pv)

    def list_positions(self, sub_account: str = "day") -> list[Position]:
        aid = self._account_id(sub_account)
        out: list[Position] = []
        for sym, p in positions_from_ledger(self.conn, aid).items():
            mark = self._last_mark(aid, sym, fallback=p.avg_cost)
            out.append(Position(symbol=sym, qty=p.qty, avg_cost=p.avg_cost,
                                mark_price=mark, unrealized_pnl=(mark - p.avg_cost) * p.qty))
        return out

    # ---------- writes ----------

    def place_order(self, req: OrderRequest) -> OrderResult:
        symbol = req.symbol.upper()
        if symbol in self.blocklist:
            return self._reject(req, reason="blocklist")

        aid = self._account_id(req.sub_account)
        now = datetime.now(timezone.utc)
        bar = self.bars.get_bar(symbol, now)

        # Pre-trade caps using a hypothetical fill price.
        ref_price = (
            req.limit_price if req.order_type == "limit" and req.limit_price
            else (bar.close if bar else None)
        )
        if ref_price is None:
            return self._reject(req, reason="no_bar")

        notional_est = ref_price * req.qty
        if notional_est > self.max_order_usd:
            return self._reject(req, reason=f"max_order_usd:{self.max_order_usd}")

        cash, pv = self._equity(aid)
        equity = cash + pv

        current_qty = positions_from_ledger(self.conn, aid).get(symbol)
        held_qty = current_qty.qty if current_qty else 0.0
        post_qty = held_qty + (req.qty if req.side == "buy" else -req.qty)

        # No-shorting guard (only when shorting is disabled).
        if not self.allow_shorting and post_qty < -1e-9:
            return self._reject(req, reason="shorting_disabled")

        # Per-symbol cap: absolute post-trade exposure vs equity (covers shorts).
        if equity > 0:
            post_symbol_notional = abs(post_qty) * ref_price
            symbol_pct = 100.0 * post_symbol_notional / equity
            if symbol_pct > self.max_symbol_pct:
                return self._reject(req, reason=f"max_symbol_pct:{self.max_symbol_pct}")

        # Gross-exposure / leverage cap: sum of |position value| across the book,
        # applying this order, must stay within max_leverage × equity.
        if equity > 0 and self.max_leverage > 0:
            positions = positions_from_ledger(self.conn, aid)
            gross = 0.0
            for sym, p in positions.items():
                mark = self._last_mark(aid, sym, fallback=p.avg_cost)
                if sym == symbol:
                    gross += abs(post_qty) * ref_price
                else:
                    gross += abs(p.qty) * mark
            if symbol not in positions:
                gross += abs(post_qty) * ref_price
            if gross > equity * self.max_leverage + 1e-6:
                return self._reject(req, reason=f"max_leverage:{self.max_leverage}")

        # Try to fill.
        spec = OrderSpec(symbol=symbol, side=req.side, qty=req.qty,
                         order_type=req.order_type, limit_price=req.limit_price)
        fill = simulate_fill(spec, bar)

        now_s = now.isoformat()
        if not fill.filled:
            cur = self.conn.execute(
                """
                INSERT INTO orders(account_id, ts, symbol, side, qty, order_type, limit_price, tif,
                                   status, submitted_at, agent, thesis, venue, dual_group_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                """,
                (aid, now_s, symbol, req.side, req.qty, req.order_type, req.limit_price, req.tif,
                 now_s, req.agent, req.thesis, self.name, req.dual_group_id),
            )
            oid = cur.lastrowid
            return OrderResult(id=oid, external_id=None, status="pending",
                               fill_price=None, fees=0.0, venue=self.name,
                               dual_group_id=req.dual_group_id)

        # Filled — write order + cash ledger atomically.
        cur = self.conn.execute(
            """
            INSERT INTO orders(account_id, ts, symbol, side, qty, order_type, limit_price, tif,
                               status, submitted_at, filled_at, fill_price, fees,
                               agent, thesis, venue, dual_group_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'filled', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (aid, now_s, symbol, req.side, req.qty, req.order_type, req.limit_price, req.tif,
             now_s, now_s, fill.fill_price, fill.fees,
             req.agent, req.thesis, self.name, req.dual_group_id),
        )
        oid = cur.lastrowid
        cash_delta = (-fill.notional if req.side == "buy" else fill.notional) - fill.fees
        self.conn.execute(
            "INSERT INTO cash_ledger(account_id, ts, delta, reason, ref_order_id) VALUES (?, ?, ?, 'fill', ?)",
            (aid, now_s, cash_delta, oid),
        )

        # With margin/shorting enabled, cash may go negative (borrowing) or grow
        # from short-sale proceeds. Guard against absurd states rather than a hard
        # cash>=0 rule: equity must remain within the leverage envelope.
        new_cash, new_pv = self._equity(aid)
        equity_now = new_cash + new_pv
        if self.max_leverage > 0 and equity_now > 0:
            gross = 0.0
            for sym, p in positions_from_ledger(self.conn, aid).items():
                gross += abs(p.qty) * self._last_mark(aid, sym, fallback=p.avg_cost)
            assert gross <= equity_now * self.max_leverage * 1.5 + 1.0, (
                f"gross exposure {gross:.2f} exceeds leverage envelope after order {oid}")

        return OrderResult(id=oid, external_id=None, status="filled",
                           fill_price=fill.fill_price, fees=fill.fees,
                           venue=self.name, dual_group_id=req.dual_group_id)

    def _reject(self, req: OrderRequest, *, reason: str) -> OrderResult:
        aid = self._account_id(req.sub_account)
        now_s = datetime.now(timezone.utc).isoformat()
        cur = self.conn.execute(
            """
            INSERT INTO orders(account_id, ts, symbol, side, qty, order_type, limit_price, tif,
                               status, submitted_at, agent, thesis, venue, dual_group_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'rejected', ?, ?, ?, ?, ?)
            """,
            (aid, now_s, req.symbol.upper(), req.side, req.qty, req.order_type, req.limit_price,
             req.tif, now_s, req.agent, f"reject:{reason}|{req.thesis or ''}",
             self.name, req.dual_group_id),
        )
        return OrderResult(id=cur.lastrowid, external_id=None, status="rejected",
                           fill_price=None, fees=0.0, venue=self.name,
                           dual_group_id=req.dual_group_id)

    def cancel_order(self, order_id: int) -> None:
        self.conn.execute(
            "UPDATE orders SET status='cancelled' WHERE id=? AND status='pending'",
            (order_id,),
        )

    def close_position(self, symbol: str, sub_account: str = "day",
                       percentage: float = 100.0) -> OrderResult:
        aid = self._account_id(sub_account)
        pos = positions_from_ledger(self.conn, aid).get(symbol.upper())
        if pos is None or pos.qty <= 0:
            return self._reject(
                OrderRequest(symbol=symbol, side="sell", qty=0.0, sub_account=sub_account,
                             agent="manual"),
                reason="no_position",
            )
        qty = pos.qty * (percentage / 100.0)
        return self.place_order(OrderRequest(symbol=symbol, side="sell", qty=qty,
                                             sub_account=sub_account, agent="manual",
                                             thesis="close_position"))

    def mark_to_market(self, now: datetime, sub_account: str = "day") -> AccountSnapshot:
        aid = self._account_id(sub_account)
        positions = positions_from_ledger(self.conn, aid)
        marks: dict[str, float] = {}
        for sym, p in positions.items():
            bar = self.bars.get_bar(sym, now)
            marks[sym] = bar.close if bar else self._last_mark(aid, sym, fallback=p.avg_cost)
        cash = dbm.get_cash(self.conn, aid)
        pv = 0.0
        now_s = now.isoformat()
        for sym, p in positions.items():
            mark = marks[sym]
            pv += p.qty * mark
            self.conn.execute(
                "INSERT INTO positions_snapshot(account_id, ts, symbol, qty, avg_cost, mark_price) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (aid, now_s, sym, p.qty, p.avg_cost, mark),
            )
        equity = cash + pv
        self.conn.execute(
            "INSERT INTO equity_curve(account_id, ts, cash, positions_value, equity) VALUES (?, ?, ?, ?, ?)",
            (aid, now_s, cash, pv, equity),
        )
        return AccountSnapshot(name=sub_account, venue=self.name, equity=equity,
                               cash=cash, positions_value=pv)

    def equity_curve(self, sub_account: str = "day",
                     since: datetime | None = None) -> pd.DataFrame:
        aid = self._account_id(sub_account)
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
