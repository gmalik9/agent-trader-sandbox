"""Typed wrappers around the short-term-trader MCP server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import get_settings
from src.mcp_clients.base import MCPClient

_ENV_PASSTHROUGH = (
    "ALPACA_API_KEY_ID", "ALPACA_SECRET_KEY", "ALPACA_PAPER",
    "FINNHUB_API_KEY", "ALPHAVANTAGE_API_KEY",
    "MARKETAUX_API_KEY", "NEWSAPI_KEY", "TIINGO_API_KEY",
)


class ShortTermClient:
    def __init__(self, cwd: str | Path | None = None, **kw: Any) -> None:
        s = get_settings()
        path = cwd or s.short_term_trader_path
        if not path:
            raise ValueError("SHORT_TERM_TRADER_PATH not set")
        self._mcp = MCPClient(cwd=Path(path), module="mcp_server.server",
                              env_passthrough=_ENV_PASSTHROUGH, **kw)

    # lifecycle
    def start(self) -> None: self._mcp.start()
    def stop(self) -> None: self._mcp.stop()
    def health(self) -> bool: return self._mcp.health()
    def list_tool_names(self) -> list[str]: return self._mcp.list_tool_names()

    # ---------- tools (exact upstream names) ----------

    def market_status(self) -> dict:
        return self._mcp.call("market_status")

    def scan_latest(self, *, universe: str = "liquid", limit: int = 25) -> dict:
        return self._mcp.call("scan_latest", universe=universe, limit=limit)

    def scan_run(self, *, mode: str = "intraday", universe: str = "liquid",
                 watchlist: list[str] | None = None) -> dict:
        return self._mcp.call("scan_run", mode=mode, universe=universe,
                              watchlist=watchlist or [])

    def list_ideas(self, *, mode: str = "intraday", tier: str = "A",
                   limit: int = 10) -> dict:
        return self._mcp.call("list_ideas", mode=mode, tier=tier, limit=limit)

    def lookup_ticker(self, ticker: str, *, interval: str = "5m",
                      period: str = "5d") -> dict:
        return self._mcp.call("lookup_ticker", ticker=ticker, interval=interval, period=period)

    def get_news(self, ticker: str, *, days: int = 1, limit: int = 20) -> dict:
        return self._mcp.call("get_news", ticker=ticker, days=days, limit=limit)


EXPECTED_TOOLS = {"market_status", "scan_latest", "scan_run", "list_ideas",
                  "lookup_ticker", "get_news"}
