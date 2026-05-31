"""Sandbox fill engine and mark-to-market.

Fill model (v1):
- Market orders fill at the *bar close* of the bar containing `now`, plus
  slippage in the trade direction (`+slippage_bps` on buy, `-slippage_bps` on sell).
- Limit orders fill at `limit_price` iff `bar.low <= limit_price <= bar.high`;
  otherwise the order is left `pending`.
- Commission = `commission_bps * notional`.
- Fractional shares allowed.

Bar source: yfinance 1-minute bars, cached per (symbol, date) on disk under
`data/bars/`. Callers may inject a `bar_provider` for tests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd

from src.config import DATA_DIR, get_settings

log = logging.getLogger(__name__)

BARS_DIR = DATA_DIR / "bars"


@dataclass
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class BarProvider(Protocol):
    def get_bar(self, symbol: str, ts: datetime) -> Bar | None: ...


@dataclass
class FillResult:
    filled: bool
    fill_price: float | None
    fees: float
    notional: float
    reason: str = ""


@dataclass
class OrderSpec:
    symbol: str
    side: str  # 'buy' | 'sell'
    qty: float
    order_type: str = "market"  # 'market' | 'limit'
    limit_price: float | None = None


class YFinanceBarProvider:
    """1-minute bars from yfinance, on-disk cache per (symbol, UTC date)."""

    def __init__(self, cache_dir: Path = BARS_DIR) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[tuple[str, str], pd.DataFrame] = {}

    def _cache_path(self, symbol: str, day: pd.Timestamp) -> Path:
        return self.cache_dir / f"{symbol.upper()}_{day.strftime('%Y-%m-%d')}.parquet"

    def _load_day(self, symbol: str, day: pd.Timestamp) -> pd.DataFrame:
        key = (symbol.upper(), day.strftime("%Y-%m-%d"))
        if key in self._mem:
            return self._mem[key]
        path = self._cache_path(symbol, day)
        if path.exists():
            df = pd.read_parquet(path)
        else:
            import yfinance as yf

            start = day.tz_localize(None)
            end = start + pd.Timedelta(days=1)
            df = yf.download(
                symbol,
                interval="1m",
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                df = df.tz_convert("UTC") if df.index.tz else df.tz_localize("UTC")
                df.to_parquet(path)
        self._mem[key] = df
        return df

    def get_bar(self, symbol: str, ts: datetime) -> Bar | None:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        day = pd.Timestamp(ts).tz_convert("UTC").normalize()
        df = self._load_day(symbol, day)
        if df.empty:
            return None
        target = pd.Timestamp(ts).tz_convert("UTC")
        idx = df.index.get_indexer([target], method="pad")[0]
        if idx < 0:
            return None
        row = df.iloc[idx]
        return Bar(
            ts=row.name.to_pydatetime(),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]),
        )


def simulate_fill(
    order: OrderSpec,
    bar: Bar | None,
    *,
    slippage_bps: float | None = None,
    commission_bps: float | None = None,
) -> FillResult:
    if bar is None:
        return FillResult(filled=False, fill_price=None, fees=0.0, notional=0.0, reason="no_bar")

    s = get_settings()
    slip = slippage_bps if slippage_bps is not None else s.slippage_bps
    comm = commission_bps if commission_bps is not None else s.commission_bps

    if order.side not in ("buy", "sell"):
        raise ValueError(f"bad side: {order.side}")
    if order.qty <= 0:
        return FillResult(False, None, 0.0, 0.0, "non_positive_qty")

    if order.order_type == "market":
        base = bar.close
        slip_mult = 1 + slip / 10_000 if order.side == "buy" else 1 - slip / 10_000
        price = base * slip_mult
    elif order.order_type == "limit":
        if order.limit_price is None:
            raise ValueError("limit order requires limit_price")
        if not (bar.low <= order.limit_price <= bar.high):
            return FillResult(False, None, 0.0, 0.0, "limit_not_touched")
        price = order.limit_price
    else:
        raise ValueError(f"unsupported order_type: {order.order_type}")

    notional = price * order.qty
    fees = abs(notional) * (comm / 10_000)
    return FillResult(filled=True, fill_price=price, fees=fees, notional=notional)


# ---------- mark-to-market ----------

@dataclass
class Position:
    symbol: str
    qty: float
    avg_cost: float


def positions_from_ledger(conn, account_id: int) -> dict[str, Position]:
    """Reconstruct current positions from filled orders for this account."""
    rows = conn.execute(
        """
        SELECT symbol, side, qty, fill_price
          FROM orders
         WHERE account_id = ? AND status = 'filled'
         ORDER BY ts ASC
        """,
        (account_id,),
    ).fetchall()
    pos: dict[str, Position] = {}
    for r in rows:
        sym = r["symbol"]
        signed = r["qty"] if r["side"] == "buy" else -r["qty"]
        p = pos.get(sym)
        if p is None:
            pos[sym] = Position(symbol=sym, qty=signed, avg_cost=float(r["fill_price"] or 0.0))
            continue
        new_qty = p.qty + signed
        if (p.qty >= 0 and signed >= 0) or (p.qty <= 0 and signed <= 0):
            # adding to the position — weighted-avg cost
            if new_qty != 0:
                p.avg_cost = (p.avg_cost * p.qty + float(r["fill_price"]) * signed) / new_qty
            p.qty = new_qty
        else:
            # reducing / flipping
            p.qty = new_qty
            if new_qty == 0:
                p.avg_cost = 0.0
            elif (p.qty > 0) != (signed > 0):
                # flipped sign — reset basis to this fill
                p.avg_cost = float(r["fill_price"])
    return {s: p for s, p in pos.items() if abs(p.qty) > 1e-9}
