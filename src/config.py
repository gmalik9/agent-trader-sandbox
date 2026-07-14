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
    llm_model: str = Field(default="openai/gpt-5")
    github_token: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    # Rate-limit resilience: on HTTP 429 the provider retries with exponential
    # backoff (respecting the server's Retry-After header, capped) before giving
    # up and letting the adaptive provider downshift.
    llm_max_retries: int = 3
    llm_retry_backoff: float = 1.5      # seconds, base for exponential backoff
    llm_retry_cap: float = 8.0          # max seconds to wait on any single retry
    # Day-trader cadence. Lower = higher frequency (bounded by the LLM's rate
    # limit). 60s is the safe default for the GitHub Models free tier.
    day_tick_seconds: int = 60
    # Alpaca leg is the source of truth — retry transient order failures
    # (spurious trading_disabled, transport errors) before giving up.
    alpaca_max_retries: int = 4
    alpaca_retry_backoff: float = 1.0   # seconds, base for exponential backoff
    alpaca_retry_cap: float = 6.0       # max seconds per retry wait

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

    # Day-agent risk discipline. Independent of the (optional) venue caps above,
    # the day agent never concentrates more than this share of equity into a
    # single symbol per trade. Prevents cheap/volatile names (e.g. a $4 leveraged
    # ETF) from consuming the whole account under 1%-risk sizing. Set to 0 to
    # disable (not recommended).
    day_max_position_pct: float = 0.20   # ≤20% of equity per single-name entry
    # Diversification enforcement (mechanical, not just prompt guidance):
    #  - theme cap: total exposure to one correlation theme (e.g. all semis:
    #    SOXL/SOXS/NVDL) can't exceed this share of equity — stops the agent
    #    loading the book with correlated names.
    #  - cooldown: a name opened/added in the last N seconds is hidden from the
    #    idea list, forcing the agent to rotate into DIFFERENT names each tick.
    day_theme_max_pct: float = 0.35      # ≤35% of equity per correlation theme
    day_name_cooldown_seconds: int = 600  # 10 min per-name cooldown (0 disables)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Merge secrets.toml into os.environ (env wins if already set).
    for k, v in _load_secrets_file().items():
        os.environ.setdefault(k, v)
    return Settings()  # type: ignore[call-arg]


def db_path() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / "sandbox.sqlite"
