"""Direct Alpaca **paper** options client (calls & puts).

The upstream stock-recommender MCP is equities-only, so options orders go
through this thin REST client instead. It talks *only* to Alpaca's paper API
and replicates the same defense-in-depth guarantees as the MCP broker:

  1. Base URL is hard-coded to https://paper-api.alpaca.markets — there is no
     flag to switch to live trading.
  2. `ALPACA_PAPER` must be the literal string "true".
  3. The account_number returned by /v2/account must start with "PA" (Alpaca's
     paper-account prefix) — verified before any write.

Options are still "third-party tracked" on Alpaca exactly like the equity
trades, and every order is mirrored into our local `orders` table for the
dashboard + audit, with the agent's reasoning logged in `agent_runs` /
`reasoning_log.jsonl` as usual.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

import httpx

from src.config import get_settings
from src.sandbox import db as dbm

log = logging.getLogger(__name__)

PAPER_BASE_URL = "https://paper-api.alpaca.markets"


class OptionsSafetyError(RuntimeError):
    """Raised when a request would breach the paper-only guarantees."""


class AlpacaOptions:
    """Minimal paper-only Alpaca options client."""

    def __init__(self, *, timeout: float = 15.0) -> None:
        s = get_settings()
        self._key = s.alpaca_api_key_id
        self._secret = s.alpaca_secret_key
        self._paper_flag = (s.alpaca_paper or "false").strip().lower()
        self._timeout = timeout
        self._verified = False

    # ---------- safety ----------

    def _assert_paper(self) -> None:
        if self._paper_flag != "true":
            raise OptionsSafetyError("ALPACA_PAPER must be the literal string 'true'.")
        if not self._key or not self._secret:
            raise OptionsSafetyError("Alpaca paper credentials missing.")

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._key,
            "APCA-API-SECRET-KEY": self._secret,
            "Accept": "application/json",
        }

    def _request(self, method: str, path: str, **kw: Any) -> Any:
        self._assert_paper()
        url = f"{PAPER_BASE_URL}{path}"
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.request(method, url, headers=self._headers(), **kw)
        if resp.status_code >= 400:
            raise OptionsSafetyError(
                f"alpaca {method} {path} -> {resp.status_code}: {resp.text[:300]}")
        return resp.json() if resp.content else {}

    def _verify_paper_account(self) -> None:
        if self._verified:
            return
        acct = self._request("GET", "/v2/account")
        acct_no = str(acct.get("account_number") or "")
        if not acct_no.startswith("PA"):
            raise OptionsSafetyError(
                f"Refusing: account_number {acct_no!r} is not a paper account (expected 'PA').")
        self._verified = True

    # ---------- reads ----------

    def get_account(self) -> dict:
        return self._request("GET", "/v2/account")

    def options_enabled(self) -> bool:
        try:
            acct = self.get_account()
            return int(acct.get("options_trading_level", 0) or 0) >= 1
        except Exception:
            return False

    def find_contracts(self, underlying: str, *, option_type: str = "call",
                       limit: int = 20, expiration_gte: str | None = None,
                       expiration_lte: str | None = None,
                       strike_gte: float | None = None,
                       strike_lte: float | None = None) -> list[dict]:
        """List tradable option contracts for an underlying.

        `option_type` is 'call' or 'put'. Returns simplified contract dicts.
        """
        params: dict[str, Any] = {
            "underlying_symbols": underlying.upper(),
            "type": "call" if option_type.lower().startswith("c") else "put",
            "status": "active",
            "limit": min(int(limit), 100),
        }
        if expiration_gte:
            params["expiration_date_gte"] = expiration_gte
        if expiration_lte:
            params["expiration_date_lte"] = expiration_lte
        if strike_gte is not None:
            params["strike_price_gte"] = str(strike_gte)
        if strike_lte is not None:
            params["strike_price_lte"] = str(strike_lte)
        data = self._request("GET", "/v2/options/contracts", params=params)
        contracts = data.get("option_contracts", data.get("contracts", [])) or []
        out = []
        for c in contracts:
            out.append({
                "symbol": c.get("symbol"),
                "underlying": c.get("underlying_symbol"),
                "type": c.get("type"),
                "strike": float(c.get("strike_price", 0) or 0),
                "expiration": c.get("expiration_date"),
                "style": c.get("style"),
                "open_interest": c.get("open_interest"),
                "close_price": c.get("close_price"),
            })
        return out

    def list_positions(self) -> list[dict]:
        """All positions; option positions have asset_class 'us_option'."""
        try:
            positions = self._request("GET", "/v2/positions")
        except Exception:
            return []
        return [p for p in positions if p.get("asset_class") in ("us_option", "option")]

    # ---------- writes ----------

    def place_order(self, *, occ_symbol: str, qty: int, side: str,
                    order_type: str = "market", limit_price: float | None = None,
                    time_in_force: str = "day") -> dict:
        """Submit an options order (1 contract = 100 shares).

        `occ_symbol` is the OCC symbol, e.g. 'AAPL250620C00190000'.
        """
        self._verify_paper_account()
        body: dict[str, Any] = {
            "symbol": occ_symbol,
            "qty": str(int(qty)),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if order_type == "limit" and limit_price is not None:
            body["limit_price"] = str(limit_price)
        return self._request("POST", "/v2/orders", json=body)


class OptionsRecorder:
    """Persist option orders into the local `orders` table for the dashboard."""

    VENUE = "alpaca_options"

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def record(self, *, sub_account: str, occ_symbol: str, side: str, qty: int,
               agent: str, thesis: str | None, resp: dict | None,
               error: str | None = None) -> int:
        aid = dbm.get_account_id(
            self.conn, {"day": "day_alpaca", "long": "long_alpaca"}.get(sub_account, sub_account))
        now_s = datetime.now(timezone.utc).isoformat()
        if error is not None:
            status, ext_id, fill = "rejected", None, None
            thesis = f"option_error:{error}|{thesis or ''}"[:500]
        else:
            ext_id = str((resp or {}).get("id") or (resp or {}).get("order_id") or "")
            status = str((resp or {}).get("status", "accepted")) if ext_id else "rejected"
            fp = (resp or {}).get("filled_avg_price")
            fill = float(fp) if fp else None
            if not ext_id:
                status = "rejected"
                thesis = f"option_not_placed|{thesis or ''}"[:500]
        cur = self.conn.execute(
            """
            INSERT INTO orders(account_id, ts, symbol, side, qty, order_type, tif,
                               status, submitted_at, filled_at, fill_price, external_id,
                               agent, thesis, venue)
            VALUES (?, ?, ?, ?, ?, 'market', 'day', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (aid, now_s, occ_symbol.upper(), side, qty, status, now_s,
             now_s if status == "filled" else None, fill, ext_id or None,
             agent, thesis, self.VENUE),
        )
        return cur.lastrowid
