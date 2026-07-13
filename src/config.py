"""Centralized settings.

Resolution order (highest precedence first):
1. Explicit kwarg
2. `settings` table in SQLite
3. `.streamlit/secrets.toml`
4. Environment variable
5. Hard-coded default
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SECRETS_PATH = REPO_ROOT / ".streamlit" / "secrets.toml"


def _load_secrets_file() -> dict[str, str]:
    if not SECRETS_PATH.exists():
        return {}
    try:
        with SECRETS_PATH.open("rb") as fh:
            raw = tomllib.load(fh)
    except Exception:  # noqa: BLE001 — a malformed secrets file must not crash boot
        return {}
    return {str(k): str(v) for k, v in raw.items() if not isinstance(v, dict)}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: str = Field(default="github")
    llm_model: str = Field(default="openai/gpt-4o-mini")
    github_token: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Broker
    broker_backend: str = Field(default="dual")  # 'sandbox' | 'alpaca_paper' | 'dual'
    alpaca_api_key_id: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: str = "true"
    stock_rec_mcp_trading_enabled: str = "true"
    stock_rec_max_order_usd: str = "1000"
    stock_rec_max_symbol_pct: str = "20"
    # Which leg drives P&L / sizing in the dual broker: 'alpaca' | 'sandbox'.
    # 'alpaca' means "execute on Alpaca whenever possible" — the sandbox becomes
    # the mirror. Falls back to sandbox-only when the Alpaca MCP is unavailable.
    dual_primary: str = Field(default="alpaca")

    # Upstream paths
    short_term_trader_path: str = ""
    stock_recommender_path: str = ""

    # Capital
    capital_total: float = 100_000.0
    split_day_pct: float = 30.0

    # Optional upstream keys
    finnhub_api_key: str = ""
    alphavantage_api_key: str = ""

    # Sandbox engine knobs
    slippage_bps: float = 2.0
    commission_bps: float = 1.0

    # Trading permissions (paper only). Enabling shorting + leveraged products.
    allow_shorting: bool = True          # allow selling to open negative positions
    allow_leveraged: bool = True         # allow leveraged / inverse / vol ETFs
    max_leverage: float = 2.0            # cap gross exposure at N× account equity


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Merge secrets.toml into os.environ (env wins if already set).
    for k, v in _load_secrets_file().items():
        os.environ.setdefault(k, v)
    return Settings()  # type: ignore[call-arg]


def db_path() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / "sandbox.sqlite"
