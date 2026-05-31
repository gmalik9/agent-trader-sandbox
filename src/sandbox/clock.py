"""Market-clock helpers using pandas_market_calendars (XNYS)."""

from __future__ import annotations

from datetime import datetime, time, timezone

import pandas as pd
import pandas_market_calendars as mcal

_XNYS = mcal.get_calendar("XNYS")
_ET = "America/New_York"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _as_ts(ts: datetime | None) -> pd.Timestamp:
    if ts is None:
        ts = now_utc()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return pd.Timestamp(ts)


def is_market_open(ts: datetime | None = None) -> bool:
    t = _as_ts(ts)
    sched = _XNYS.schedule(start_date=t.normalize(), end_date=t.normalize())
    if sched.empty:
        return False
    open_, close_ = sched.iloc[0]["market_open"], sched.iloc[0]["market_close"]
    return open_ <= t <= close_


def is_force_flat_window(ts: datetime | None = None, *, minutes_before_close: int = 5) -> bool:
    """True in the last `minutes_before_close` of the regular session."""
    t = _as_ts(ts)
    sched = _XNYS.schedule(start_date=t.normalize(), end_date=t.normalize())
    if sched.empty:
        return False
    close_ = sched.iloc[0]["market_close"]
    return (close_ - pd.Timedelta(minutes=minutes_before_close)) <= t <= close_


def next_open(ts: datetime | None = None) -> datetime:
    t = _as_ts(ts)
    sched = _XNYS.schedule(start_date=t.normalize(), end_date=t.normalize() + pd.Timedelta(days=7))
    for _, row in sched.iterrows():
        if row["market_open"] > t:
            return row["market_open"].to_pydatetime()
    raise RuntimeError("no upcoming market open in next 7 days")


def next_close(ts: datetime | None = None) -> datetime:
    t = _as_ts(ts)
    sched = _XNYS.schedule(start_date=t.normalize(), end_date=t.normalize() + pd.Timedelta(days=7))
    for _, row in sched.iterrows():
        if row["market_close"] > t:
            return row["market_close"].to_pydatetime()
    raise RuntimeError("no upcoming market close in next 7 days")
