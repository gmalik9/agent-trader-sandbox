"""Self-contained fallback signal providers (yfinance-only).

When the sibling MCP servers (``short-term-trader`` / ``stock-recommender``)
aren't reachable — most importantly on Streamlit Cloud, where those repos don't
exist and their paths aren't configured — the agents would otherwise have no
source of trade ideas, quotes, or recommendations, and would place zero trades.

These two classes give the agents a real (if simpler) source built directly on
yfinance. They expose the *exact same method signatures* the agents call on the
real MCP clients (:class:`~src.mcp_clients.short_term.ShortTermClient` and
:class:`~src.mcp_clients.long_term.LongTermClient`), so the agent code doesn't
change — the scheduler just swaps in a local client when the MCP subprocess
can't be launched.

Signals are intentionally simple and transparent:
- momentum (rate of change over a lookback window)
- trend (last close vs. its 50/200-day simple moving averages)
- ATR-based stop suggestion for intraday ideas

None of this touches a broker; it only produces data for the LLM to reason over.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


# A compact, sector-diverse, liquid universe. Kept small so a single batched
# yfinance download stays fast enough to run inside a scheduler tick.
_UNIVERSE: list[str] = [
    # Information Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "AMD", "CRM", "ADBE",
    # Communication Services
    "GOOGL", "META", "NFLX", "DIS",
    # Consumer Discretionary
    "AMZN", "HD", "NKE", "MCD", "SBUX", "LOW",
    # Consumer Staples
    "PG", "KO", "COST", "WMT", "PEP",
    # Financials
    "JPM", "BAC", "V", "MA", "GS",
    # Health Care
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK",
    # Industrials
    "CAT", "HON", "RTX", "GE", "DE",
    # Energy
    "XOM", "CVX", "COP",
    # Broad ETFs
    "SPY", "QQQ", "IWM", "DIA",
]

# Leveraged / inverse / vol products are never surfaced as ideas.
_BLOCKED = {
    "TQQQ", "SQQQ", "SOXL", "SOXS", "UPRO", "SPXU", "TNA", "TZA",
    "UVXY", "SVXY", "VXX", "VIXY", "LABU", "LABD", "FAS", "FAZ",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


class _PriceCache:
    """Tiny thread-safe TTL cache over batched yfinance history downloads."""

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._store: dict[tuple[str, str], tuple[float, dict[str, pd.DataFrame]]] = {}

    def get(self, symbols: tuple[str, ...], period: str, interval: str) -> dict[str, pd.DataFrame]:
        key = (period, interval)
        with self._lock:
            hit = self._store.get(key)
            if hit and (time.time() - hit[0]) < self._ttl:
                cached = hit[1]
                if all(s in cached for s in symbols):
                    return {s: cached[s] for s in symbols}
        fresh = _download(list(symbols), period=period, interval=interval)
        with self._lock:
            existing = self._store.get(key)
            merged = dict(existing[1]) if existing and (time.time() - existing[0]) < self._ttl else {}
            merged.update(fresh)
            self._store[key] = (time.time(), merged)
        return fresh


def _download(symbols: list[str], *, period: str, interval: str) -> dict[str, pd.DataFrame]:
    """Batch-download OHLCV history and return a per-symbol frame dict."""
    if not symbols:
        return {}
    import yfinance as yf

    try:
        raw = yf.download(
            symbols,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=True,
            group_by="ticker",
        )
    except Exception as e:  # noqa: BLE001 — network hiccups must not crash a tick
        log.warning("yfinance batch download failed (%s); returning empty", e)
        return {}

    out: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for sym in symbols:
            if sym in raw.columns.get_level_values(0):
                df = raw[sym].dropna(how="all")
                if not df.empty:
                    out[sym] = df
    else:  # single symbol → flat columns
        df = raw.dropna(how="all")
        if not df.empty:
            out[symbols[0]] = df
    return out


def _atr(df: pd.DataFrame, window: int = 14) -> float:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else 0.0


def _roc(df: pd.DataFrame, window: int) -> float:
    close = df["Close"]
    if len(close) <= window:
        return 0.0
    prev = float(close.iloc[-window - 1])
    last = float(close.iloc[-1])
    if prev <= 0:
        return 0.0
    return 100.0 * (last - prev) / prev


def _sma(df: pd.DataFrame, window: int) -> float:
    close = df["Close"]
    if len(close) < window:
        return float("nan")
    return float(close.rolling(window).mean().iloc[-1])


# --------------------------------------------------------------------------- #
#  Short-term (day-trader) fallback
# --------------------------------------------------------------------------- #


class LocalShortTermClient:
    """yfinance-backed stand-in for :class:`ShortTermClient`.

    Implements the subset of methods the DayTraderAgent actually calls:
    ``list_ideas`` and ``lookup_ticker`` (plus no-op lifecycle methods).
    """

    is_local = True

    def __init__(self, universe: list[str] | None = None) -> None:
        self._universe = [s for s in (universe or _UNIVERSE) if s not in _BLOCKED]
        self._cache = _PriceCache(ttl_seconds=180.0)

    # lifecycle (no subprocess to manage)
    def start(self) -> None: return None
    def stop(self) -> None: return None
    def health(self) -> bool: return True
    def list_tool_names(self) -> list[str]: return ["list_ideas", "lookup_ticker"]

    def list_ideas(self, *, mode: str = "intraday", tier: str = "A",
                   limit: int = 10) -> dict:
        frames = self._cache.get(tuple(self._universe), period="1mo", interval="1d")
        ideas: list[dict[str, Any]] = []
        for sym, df in frames.items():
            if df is None or len(df) < 15:
                continue
            entry = float(df["Close"].iloc[-1])
            if entry <= 0:
                continue
            atr = _atr(df)
            mom5 = _roc(df, 5)
            mom10 = _roc(df, 10)
            score = round(0.6 * mom5 + 0.4 * mom10, 3)
            signal_count = int(mom5 > 0) + int(mom10 > 0) + int(entry > _sma(df, 10))
            idea_tier = "A" if signal_count >= 3 else ("B" if signal_count == 2 else "C")
            stop = round(entry - 1.5 * atr, 2) if atr > 0 else round(entry * 0.98, 2)
            ideas.append({
                "symbol": sym,
                "tier": idea_tier,
                "direction": "long",
                "score": score,
                "entry": round(entry, 2),
                "stop": stop,
                "atr": round(atr, 3),
                "momentum_5d_pct": round(mom5, 2),
                "momentum_10d_pct": round(mom10, 2),
            })
        # Only surface long setups with positive momentum, ranked by score.
        ideas = [i for i in ideas if i["score"] > 0]
        rank = {"A": 0, "B": 1, "C": 2}
        if tier in rank:
            ideas = [i for i in ideas if rank[i["tier"]] <= rank[tier]]
        ideas.sort(key=lambda i: (rank[i["tier"]], -i["score"]))
        return {"mode": mode, "as_of": _now().isoformat(), "count": len(ideas[:limit]),
                "ideas": ideas[:limit], "source": "local-yfinance"}

    def lookup_ticker(self, ticker: str, *, interval: str = "5m",
                      period: str = "5d") -> dict:
        sym = ticker.upper()
        frames = self._cache.get((sym,), period=period, interval=interval)
        df = frames.get(sym)
        if df is None or df.empty:
            # fall back to daily if the intraday window is empty (e.g. off-hours)
            frames = self._cache.get((sym,), period="1mo", interval="1d")
            df = frames.get(sym)
        if df is None or df.empty:
            return {"symbol": sym, "error": "no_data", "source": "local-yfinance"}
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
        change_pct = 100.0 * (last - prev) / prev if prev else 0.0
        return {
            "symbol": sym,
            "price": round(last, 4),
            "last": round(last, 4),
            "change_pct": round(change_pct, 3),
            "atr": round(_atr(df), 3) if len(df) >= 15 else None,
            "as_of": _now().isoformat(),
            "source": "local-yfinance",
        }

    def get_news(self, ticker: str, *, days: int = 1, limit: int = 20) -> dict:
        return {"symbol": ticker.upper(), "news": [], "source": "local-yfinance"}


# --------------------------------------------------------------------------- #
#  Long-term (investor) fallback
# --------------------------------------------------------------------------- #


class LocalLongTermClient:
    """yfinance-backed stand-in for :class:`LongTermClient`.

    Implements the READ methods the LongTermAgent calls: ``get_recommendations``,
    ``lookup_ticker`` and ``get_news``. It deliberately does **not** implement
    the Alpaca WRITE tools (``place_order`` etc.); those require the real
    upstream MCP server, so the broker's Alpaca leg is never wired to this
    fallback (the scheduler keeps them separate).
    """

    is_local = True

    def __init__(self, universe: list[str] | None = None) -> None:
        self._universe = [s for s in (universe or _UNIVERSE) if s not in _BLOCKED]
        self._cache = _PriceCache(ttl_seconds=600.0)

    # lifecycle (no subprocess to manage)
    def start(self) -> None: return None
    def stop(self) -> None: return None
    def health(self) -> bool: return True
    def list_tool_names(self) -> list[str]:
        return ["get_recommendations", "lookup_ticker", "get_news"]

    def get_recommendations(self, *, universe: str = "Curated",
                            max_per_sector: int = 3, top_n: int = 18) -> dict:
        frames = self._cache.get(tuple(self._universe), period="1y", interval="1d")
        recs: list[dict[str, Any]] = []
        for sym, df in frames.items():
            if df is None or len(df) < 60:
                continue
            price = float(df["Close"].iloc[-1])
            if price <= 0:
                continue
            mom_3m = _roc(df, 63)
            mom_6m = _roc(df, 126)
            sma50 = _sma(df, 50)
            sma200 = _sma(df, 200)
            above_50 = pd.notna(sma50) and price > sma50
            above_200 = pd.notna(sma200) and price > sma200
            score = round(0.5 * mom_6m + 0.5 * mom_3m
                          + (5.0 if above_50 else 0.0)
                          + (5.0 if above_200 else 0.0), 3)
            recs.append({
                "symbol": sym,
                "price": round(price, 2),
                "score": score,
                "momentum_3m_pct": round(mom_3m, 2),
                "momentum_6m_pct": round(mom_6m, 2),
                "above_50dma": bool(above_50),
                "above_200dma": bool(above_200),
                "rating": "buy" if score > 10 else ("hold" if score > -5 else "avoid"),
            })
        recs.sort(key=lambda r: -r["score"])
        return {"universe": universe, "as_of": _now().isoformat(),
                "recommendations": recs[:top_n], "source": "local-yfinance"}

    def get_portfolio_suggestion(self, *, budget: float = 5000.0,
                                  universe: str = "Curated", top_n: int = 18) -> dict:
        recs = self.get_recommendations(universe=universe, top_n=top_n)["recommendations"]
        buys = [r for r in recs if r["rating"] == "buy"] or recs[: max(1, top_n // 3)]
        weight = 100.0 / len(buys) if buys else 0.0
        return {
            "budget": budget,
            "as_of": _now().isoformat(),
            "suggestion": [{"symbol": r["symbol"], "target_weight_pct": round(weight, 2),
                            "price": r["price"]} for r in buys],
            "source": "local-yfinance",
        }

    def lookup_ticker(self, ticker: str) -> dict:
        sym = ticker.upper()
        frames = self._cache.get((sym,), period="1y", interval="1d")
        df = frames.get(sym)
        if df is None or df.empty:
            return {"symbol": sym, "error": "no_data", "source": "local-yfinance"}
        price = float(df["Close"].iloc[-1])
        return {
            "symbol": sym,
            "price": round(price, 4),
            "last": round(price, 4),
            "momentum_3m_pct": round(_roc(df, 63), 2),
            "momentum_6m_pct": round(_roc(df, 126), 2),
            "above_200dma": bool(pd.notna(_sma(df, 200)) and price > _sma(df, 200)),
            "as_of": _now().isoformat(),
            "source": "local-yfinance",
        }

    def get_news(self, ticker: str, *, days: int = 7, limit: int = 20) -> dict:
        return {"symbol": ticker.upper(), "news": [], "source": "local-yfinance"}


# --------------------------------------------------------------------------- #
#  Hybrid short-term client: real MCP first, local yfinance fallback
# --------------------------------------------------------------------------- #


class HybridShortTermClient:
    """Prefer the real short-term MCP scanner, fall back to local yfinance ideas.

    The upstream intraday scanner only returns ideas after a (slow, sometimes
    empty) scan; when it yields nothing or errors, the day-trader would never
    trade. This wrapper transparently substitutes the fast local momentum-based
    idea provider so the agent always has candidates to evaluate.
    """

    is_hybrid = True

    def __init__(self, real, local: LocalShortTermClient | None = None) -> None:
        self._real = real
        self._local = local or LocalShortTermClient()

    def start(self) -> None:
        try:
            self._real.start()
        except Exception:
            log.warning("hybrid: real short-term client failed to start; local only")
            self._real = None

    def stop(self) -> None:
        for c in (self._real, self._local):
            if c is not None:
                try:
                    c.stop()
                except Exception:
                    pass

    def health(self) -> bool:
        return True

    def list_tool_names(self) -> list[str]:
        return ["list_ideas", "lookup_ticker"]

    def list_ideas(self, *, mode: str = "intraday", tier: str = "A",
                   limit: int = 10) -> dict:
        if self._real is not None:
            try:
                res = self._real.list_ideas(mode=mode, tier=tier, limit=limit)
                rows = res.get("rows") or res.get("ideas") or []
                if rows:
                    return res
                log.info("hybrid: MCP scanner returned 0 ideas; using local yfinance ideas")
            except Exception as e:  # noqa: BLE001
                log.warning("hybrid: MCP list_ideas failed (%s); using local ideas", e)
        return self._local.list_ideas(mode=mode, tier=tier, limit=limit)

    def lookup_ticker(self, ticker: str, *, interval: str = "5m",
                      period: str = "5d") -> dict:
        if self._real is not None:
            try:
                res = self._real.lookup_ticker(ticker, interval=interval, period=period)
                if res and not res.get("error"):
                    return res
            except Exception:
                pass
        return self._local.lookup_ticker(ticker, interval=interval, period=period)

    def get_news(self, ticker: str, *, days: int = 1, limit: int = 20) -> dict:
        if self._real is not None:
            try:
                return self._real.get_news(ticker, days=days, limit=limit)
            except Exception:
                pass
        return self._local.get_news(ticker, days=days, limit=limit)

