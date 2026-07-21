"""Smart-money (insider + political) trade-activity signals.

Aggregates two categories of *disclosed* trading activity into a compact,
LLM-friendly signal the day agent (and the Copilot-driven skill) can use as an
extra edge:

  * **Insider (C-suite / Form 4):** officers, directors and 10% owners buying or
    selling their *own* company's stock (SEC Form 4).
  * **Political:** US Senate and House members' disclosed transactions
    (STOCK Act periodic transaction reports).

Providers are auto-selected by which API key is present:

  * **Financial Modeling Prep (FMP)** — covers BOTH insider and political with a
    single key. *Recommended.* Set ``FMP_API_KEY`` (https://financialmodelingprep.com).
    Uses the current ``/stable/`` API. NOTE: on the free plan the per-symbol
    *search* endpoints are paywalled (HTTP 402); the market-wide *latest* feeds
    are open, so this client pulls the latest insider/senate/house feeds within
    the lookback window and filters/aggregates by symbol locally.
  * **Finnhub** — insider transactions only (its congressional endpoint is a
    premium add-on). Reuses the ``FINNHUB_API_KEY`` already used by the
    news/scan legs; supports true per-symbol lookups.

If no key is configured the client degrades gracefully: every method returns a
structured ``{"available": False, ...}`` payload rather than raising, so the
agent simply treats smart-money as "no signal".

Note on "positions held": real-time full portfolios of insiders/politicians are
NOT public. What *is* public is a stream of disclosed TRANSACTIONS. This module
therefore reconstructs an APPROXIMATE net position per person / per symbol /
per sector by summing disclosed buys minus sells over a lookback window. Treat
it as "recent net activity", not a brokerage statement.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import httpx

log = logging.getLogger("smart_money")

_STABLE = "https://financialmodelingprep.com/stable"
_FINNHUB_BASE = "https://finnhub.io/api/v1"
_PAGE_SIZE = 100
# The congressional feeds cap 'limit' at 25 on the FMP free tier (larger -> 402).
_POL_PAGE_SIZE = 25

# Words in an FMP/Finnhub transaction-type string that denote an acquisition vs
# a disposition. Form 4 uses single-letter codes (P=purchase, S=sale, A=grant,
# etc.); FMP expands them into strings like "P-Purchase" / "S-Sale".
_BUY_HINTS = ("purchase", "buy", "acqui", "-p-", "p-purchase", "grant", "award")
_SELL_HINTS = ("sale", "sell", "dispos", "-s-", "s-sale")


def _classify_side(raw: str) -> str:
    """Map a free-form transaction-type string to 'buy' | 'sell' | 'other'."""
    t = (raw or "").lower()
    if any(h in t for h in _SELL_HINTS):
        return "sell"
    if any(h in t for h in _BUY_HINTS):
        return "buy"
    return "other"


def _to_float(v: Any) -> float:
    try:
        return float(str(v).replace(",", "").replace("$", "").strip() or 0)
    except (TypeError, ValueError):
        return 0.0


def _parse_day(v: Any) -> datetime | None:
    if not v:
        return None
    s = str(v).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s[: len(fmt) + 4], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _amount_midpoint(amount_range: str) -> float:
    """Congressional disclosures report a $ RANGE (e.g. '$1,001 - $15,000').

    Approximate the trade size as the midpoint of the range for aggregation.
    """
    s = (amount_range or "").replace("$", "").replace(",", "")
    nums = []
    for tok in s.replace("-", " ").split():
        try:
            nums.append(float(tok))
        except ValueError:
            continue
    if not nums:
        return 0.0
    if len(nums) == 1:
        return nums[0]
    return round((min(nums) + max(nums)) / 2.0, 2)


def _clean_symbol(v: Any) -> str:
    s = str(v or "").strip().upper()
    return "" if s in ("", "NONE", "NULL", "N/A") else s


def _norm_insider(r: dict) -> dict:
    shares = _to_float(r.get("securitiesTransacted"))
    price = _to_float(r.get("price"))
    return {
        "source": "insider",
        "symbol": _clean_symbol(r.get("symbol")),
        "person": r.get("reportingName") or r.get("typeOfOwner") or "insider",
        "role": r.get("typeOfOwner") or "",
        "side": _classify_side(r.get("transactionType") or r.get("acquisitionOrDisposition") or ""),
        "shares": shares,
        "value": round(shares * price, 2) if price else 0.0,
        "date": r.get("transactionDate") or r.get("filingDate"),
    }


def _norm_political(r: dict, chamber: str) -> dict:
    amount = str(r.get("amount") or "")
    first = str(r.get("firstName") or "").strip()
    last = str(r.get("lastName") or "").strip()
    person = (f"{first} {last}".strip()) or r.get("office") or r.get("representative") or "member"
    return {
        "source": "political",
        "chamber": chamber,
        "symbol": _clean_symbol(r.get("symbol") or r.get("ticker")),
        "person": person,
        "role": chamber,
        "side": _classify_side(r.get("type") or r.get("transactionType") or ""),
        "amount_range": amount.strip() or None,
        "value": _amount_midpoint(amount),
        # Congress discloses late, so the DISCLOSURE date (when the market
        # learns) is the actionable timestamp — use it for the lookback window.
        "date": r.get("disclosureDate") or r.get("transactionDate"),
    }


class SmartMoneyClient:
    """Fetch + summarize insider and political trading activity.

    Parameters mirror the resolved app settings; pass explicit values in tests.
    """

    def __init__(
        self,
        *,
        fmp_api_key: str = "",
        finnhub_api_key: str = "",
        lookback_days: int = 90,
        max_pages: int = 6,
        timeout: float = 12.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.fmp_key = (fmp_api_key or "").strip()
        self.finnhub_key = (finnhub_api_key or "").strip()
        self.lookback_days = max(1, int(lookback_days))
        self.max_pages = max(1, int(max_pages))
        self._timeout = float(timeout)
        self._client = client
        self._owns_client = client is None
        self._sector_cache: dict[str, str] = {}
        self._insider_cache: list[dict] | None = None
        self._political_cache: list[dict] | None = None

    # -- lifecycle -----------------------------------------------------------
    @property
    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None

    @property
    def available(self) -> bool:
        return bool(self.fmp_key or self.finnhub_key)

    @property
    def political_available(self) -> bool:
        # Only FMP exposes congressional data on the non-premium tier here.
        return bool(self.fmp_key)

    # -- low-level HTTP ------------------------------------------------------
    def _get_json(self, url: str, params: dict[str, Any]) -> Any:
        try:
            resp = self._http.get(url, params=params)
            if resp.status_code in (401, 402, 403):
                log.debug("smart_money: %s (plan/key) for %s", resp.status_code, url)
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception:  # noqa: BLE001 — a data-feed hiccup must never crash a tick
            log.debug("smart_money: request failed for %s", url, exc_info=True)
            return None

    def _fmp_get(self, path: str, params: dict[str, Any]) -> Any:
        p = dict(params)
        p["apikey"] = self.fmp_key
        return self._get_json(f"{_STABLE}/{path}", p)

    def _paginate_latest(self, path: str, normalize: Callable[[dict], dict],
                         extra: dict[str, Any] | None = None,
                         page_size: int = _PAGE_SIZE) -> list[dict]:
        """Page through a market-wide '/latest' feed until the lookback cutoff.

        ``page_size`` differs per feed: insider allows 100 rows/page but the
        congressional feeds cap ``limit`` at 25 on the FMP free tier (a larger
        value returns HTTP 402).
        """
        if not self.fmp_key:
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        rows: list[dict] = []
        for page in range(self.max_pages):
            params: dict[str, Any] = {"page": page, "limit": page_size}
            if extra:
                params.update(extra)
            data = self._fmp_get(path, params)
            if not isinstance(data, list) or not data:
                break
            kept_this_page = 0
            for r in data:
                nr = normalize(r)
                d = _parse_day(nr.get("date"))
                if d is not None and d < cutoff:
                    continue
                rows.append(nr)
                kept_this_page += 1
            # Feeds aren't always strictly date-sorted (disclosure lag), so only
            # stop once an ENTIRE page falls outside the window, or it's short.
            if kept_this_page == 0 or len(data) < page_size:
                break
        return rows

    # -- cached market-wide feeds -------------------------------------------
    def _all_insider(self) -> list[dict]:
        if self._insider_cache is None:
            self._insider_cache = self._paginate_latest(
                "insider-trading/latest", _norm_insider)
        return self._insider_cache

    def _all_political(self) -> list[dict]:
        if self._political_cache is None:
            rows: list[dict] = []
            rows += self._paginate_latest(
                "senate-latest", lambda r: _norm_political(r, "senate"), page_size=_POL_PAGE_SIZE)
            rows += self._paginate_latest(
                "house-latest", lambda r: _norm_political(r, "house"), page_size=_POL_PAGE_SIZE)
            self._political_cache = rows
        return self._political_cache

    def _finnhub_insider(self, symbol: str) -> list[dict]:
        if not self.finnhub_key or not symbol:
            return []
        frm = (datetime.now(timezone.utc) - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        to = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        data = self._get_json(
            f"{_FINNHUB_BASE}/stock/insider-transactions",
            {"symbol": symbol.upper(), "from": frm, "to": to, "token": self.finnhub_key},
        ) or {}
        out: list[dict] = []
        for r in (data.get("data") or []):
            shares = _to_float(r.get("share") or r.get("change"))
            change = _to_float(r.get("change"))
            price = _to_float(r.get("transactionPrice"))
            out.append({
                "source": "insider",
                "symbol": symbol.upper(),
                "person": r.get("name") or "insider",
                "role": "",
                "side": "buy" if change > 0 else ("sell" if change < 0 else _classify_side(r.get("transactionCode") or "")),
                "shares": abs(shares),
                "value": round(abs(shares) * price, 2) if price else 0.0,
                "date": r.get("transactionDate") or r.get("filingDate"),
            })
        return self._within_lookback(out)

    # -- aggregation helpers -------------------------------------------------
    def _within_lookback(self, rows: list[dict]) -> list[dict]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        keep = []
        for r in rows:
            d = _parse_day(r.get("date"))
            if d is None or d >= cutoff:
                keep.append(r)
        return keep

    @staticmethod
    def _summarize_side(rows: list[dict]) -> dict:
        buys = [r for r in rows if r["side"] == "buy"]
        sells = [r for r in rows if r["side"] == "sell"]
        buy_val = round(sum(_to_float(r.get("value")) for r in buys), 2)
        sell_val = round(sum(_to_float(r.get("value")) for r in sells), 2)
        net = round(buy_val - sell_val, 2)
        if buy_val or sell_val:
            bias = "bullish" if net > 0 else ("bearish" if net < 0 else "mixed")
        else:
            bias = "neutral"
        names = []
        seen = set()
        for r in sorted(rows, key=lambda x: _to_float(x.get("value")), reverse=True):
            p = str(r.get("person") or "").strip()
            if p and p.lower() not in seen:
                seen.add(p.lower())
                names.append({"person": p, "side": r["side"], "role": r.get("role") or r.get("chamber") or ""})
            if len(names) >= 5:
                break
        return {
            "buys": len(buys), "sells": len(sells),
            "buy_value": buy_val, "sell_value": sell_val,
            "net_value": net, "bias": bias, "notable": names,
        }

    # -- public API ----------------------------------------------------------
    def symbol_activity(self, symbol: str) -> dict:
        """Combined insider + political activity summary for one symbol.

        Returns a compact dict with per-category bias (bullish/bearish),
        net disclosed dollar flow, and the notable people involved. On the FMP
        free tier this is filtered from the recent market-wide feeds, so a
        thinly-traded symbol may legitimately show no recent activity.
        """
        sym = (symbol or "").upper()
        if not self.available:
            return {"available": False, "symbol": sym, "reason": "no_api_key"}
        if self.fmp_key:
            insider_rows = [r for r in self._all_insider() if r["symbol"] == sym]
        else:
            insider_rows = self._finnhub_insider(sym)
        political_rows = [r for r in self._all_political() if r["symbol"] == sym] if self.political_available else []
        ins = self._summarize_side(insider_rows)
        pol = self._summarize_side(political_rows)

        combined_net = ins["net_value"] + pol["net_value"]
        if ins["bias"] == "neutral" and pol["bias"] == "neutral":
            overall = "none"
        elif combined_net > 0:
            overall = "bullish"
        elif combined_net < 0:
            overall = "bearish"
        else:
            overall = "mixed"
        return {
            "available": True,
            "symbol": sym,
            "lookback_days": self.lookback_days,
            "overall_bias": overall,
            "insider": ins,
            "political": pol if self.political_available else {"available": False, "reason": "political_needs_fmp_key"},
        }

    def market_activity(self, *, limit: int = 25) -> dict:
        """Recent market-wide insider + political activity, grouped by symbol.

        Answers "which stocks have C-suite / political activity right now?".
        Ranks symbols by absolute net disclosed dollar flow over the lookback.
        """
        if not self.available:
            return {"available": False, "reason": "no_api_key", "symbols": []}
        rows = list(self._all_insider()) if self.fmp_key else []
        if self.political_available:
            rows += self._all_political()
        by_sym: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r.get("symbol"):
                by_sym[r["symbol"]].append(r)
        ranked = []
        for sym, srows in by_sym.items():
            summ = self._summarize_side(srows)
            has_pol = any(r["source"] == "political" for r in srows)
            has_ins = any(r["source"] == "insider" for r in srows)
            ranked.append({
                "symbol": sym,
                "net_value": summ["net_value"],
                "bias": summ["bias"],
                "buys": summ["buys"], "sells": summ["sells"],
                "has_insider": has_ins, "has_political": has_pol,
                "notable": summ["notable"][:3],
            })
        ranked.sort(key=lambda x: abs(x["net_value"]), reverse=True)
        return {
            "available": True,
            "lookback_days": self.lookback_days,
            "political_included": self.political_available,
            "symbols": ranked[: max(1, int(limit))],
        }

    def person_positions(self, name: str, *, limit: int = 40) -> dict:
        """Approximate net positions a given insider/politician has built up.

        Aggregates that person's disclosed transactions (market-wide) over the
        lookback into net buy/sell dollar flow per symbol and per sector. This
        is reconstructed from disclosures, NOT a real holdings statement.
        """
        if not self.available:
            return {"available": False, "reason": "no_api_key", "positions": []}
        needle = (name or "").strip().lower()
        if not needle:
            return {"available": True, "positions": [], "reason": "empty_name"}
        rows = list(self._all_insider()) if self.fmp_key else []
        if self.political_available:
            rows += self._all_political()
        mine = [r for r in rows if needle in str(r.get("person") or "").lower()]
        by_sym: dict[str, dict] = {}
        by_sector: dict[str, float] = defaultdict(float)
        for r in mine:
            sym = r.get("symbol")
            if not sym:
                continue
            signed = _to_float(r.get("value")) * (1 if r["side"] == "buy" else (-1 if r["side"] == "sell" else 0))
            slot = by_sym.setdefault(sym, {"symbol": sym, "net_value": 0.0, "buys": 0, "sells": 0})
            slot["net_value"] = round(slot["net_value"] + signed, 2)
            slot["buys"] += 1 if r["side"] == "buy" else 0
            slot["sells"] += 1 if r["side"] == "sell" else 0
            by_sector[self._sector_of(sym)] += signed
        positions = sorted(by_sym.values(), key=lambda x: abs(x["net_value"]), reverse=True)
        sectors = [
            {"sector": k, "net_value": round(v, 2)}
            for k, v in sorted(by_sector.items(), key=lambda kv: abs(kv[1]), reverse=True)
        ]
        return {
            "available": True,
            "person": name,
            "lookback_days": self.lookback_days,
            "transactions": len(mine),
            "positions": positions[: max(1, int(limit))],
            "by_sector": sectors,
        }

    def _sector_of(self, symbol: str) -> str:
        """Best-effort sector lookup (FMP company profile), cached in-memory."""
        sym = (symbol or "").upper()
        if not sym:
            return "Unknown"
        if sym in self._sector_cache:
            return self._sector_cache[sym]
        sector = "Unknown"
        if self.fmp_key:
            data = self._fmp_get("profile", {"symbol": sym})
            if isinstance(data, list) and data:
                sector = str(data[0].get("sector") or "Unknown") or "Unknown"
        self._sector_cache[sym] = sector
        return sector
