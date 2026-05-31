"""Typed wrappers around the stock-recommender MCP server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import get_settings
from src.mcp_clients.base import MCPClient

_ENV_PASSTHROUGH = (
    "ALPACA_API_KEY_ID", "ALPACA_SECRET_KEY", "ALPACA_PAPER",
    "STOCK_REC_MCP_TRADING_ENABLED",
    "STOCK_REC_MAX_ORDER_USD", "STOCK_REC_MAX_SYMBOL_PCT",
    "FINNHUB_API_KEY", "ALPHAVANTAGE_API_KEY",
    "MARKETAUX_API_KEY", "NEWSAPI_KEY", "TIINGO_API_KEY",
)


class LongTermClient:
    def __init__(self, cwd: str | Path | None = None, **kw: Any) -> None:
        s = get_settings()
        path = cwd or s.stock_recommender_path
        if not path:
            raise ValueError("STOCK_RECOMMENDER_PATH not set")
        self._mcp = MCPClient(cwd=Path(path), module="mcp_server.server",
                              env_passthrough=_ENV_PASSTHROUGH, **kw)

    # lifecycle
    def start(self) -> None: self._mcp.start()
    def stop(self) -> None: self._mcp.stop()
    def health(self) -> bool: return self._mcp.health()
    def list_tool_names(self) -> list[str]: return self._mcp.list_tool_names()

    # ---------- READ tools ----------

    def get_recommendations(self, *, universe: str = "Curated",
                            max_per_sector: int = 3, top_n: int = 18) -> dict:
        return self._mcp.call("get_recommendations", universe=universe,
                              max_per_sector=max_per_sector, top_n=top_n)

    def get_portfolio_suggestion(self, *, budget: float = 5000.0,
                                  universe: str = "Curated", top_n: int = 18) -> dict:
        return self._mcp.call("get_portfolio_suggestion", budget=budget,
                              universe=universe, top_n=top_n)

    def lookup_ticker(self, ticker: str) -> dict:
        return self._mcp.call("lookup_ticker", ticker=ticker)

    def get_news(self, ticker: str, *, days: int = 7, limit: int = 20) -> dict:
        return self._mcp.call("get_news", ticker=ticker, days=days, limit=limit)

    def get_account(self) -> dict:
        return self._mcp.call("get_account")

    def list_positions(self) -> list[dict]:
        out = self._mcp.call("list_positions")
        return out if isinstance(out, list) else out.get("positions", [])

    def list_orders(self, *, status: str = "open", limit: int = 50) -> list[dict]:
        out = self._mcp.call("list_orders", status=status, limit=limit)
        return out if isinstance(out, list) else out.get("orders", [])

    # ---------- WRITE tools (require STOCK_REC_MCP_TRADING_ENABLED=true) ----------

    def place_order(self, *, symbol: str, qty: float, side: str,
                    order_type: str = "market", time_in_force: str = "day",
                    limit_price: float | None = None) -> dict:
        args: dict[str, Any] = {
            "symbol": symbol, "qty": qty, "side": side,
            "order_type": order_type, "time_in_force": time_in_force,
        }
        if limit_price is not None:
            args["limit_price"] = limit_price
        return self._mcp.call("place_order", **args)

    def cancel_order(self, order_id: str) -> dict:
        return self._mcp.call("cancel_order", order_id=order_id)

    def cancel_all_orders(self) -> dict:
        return self._mcp.call("cancel_all_orders")

    def close_position(self, symbol: str, percentage: float = 100.0) -> dict:
        return self._mcp.call("close_position", symbol=symbol, percentage=percentage)

    def rebalance_to_recommendations(self, *, budget: float = 5000.0,
                                      universe: str = "Curated", top_n: int = 18,
                                      dry_run: bool = True,
                                      cash_buffer_pct: float = 5.0) -> dict:
        return self._mcp.call("rebalance_to_recommendations",
                              budget=budget, universe=universe, top_n=top_n,
                              dry_run=dry_run, cash_buffer_pct=cash_buffer_pct)


EXPECTED_TOOLS = {
    "get_recommendations", "get_portfolio_suggestion", "lookup_ticker", "get_news",
    "get_account", "list_positions", "list_orders",
    "place_order", "cancel_order", "cancel_all_orders", "close_position",
    "rebalance_to_recommendations",
}
