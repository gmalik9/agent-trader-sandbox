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
    # Intraday scanner refresh. list_ideas only returns the LAST scan_run result,
    # so without a periodic refresh the idea universe (and its prices) goes stale.
    # scan_run rescans the universe and shares the sibling MCP subprocess with
    # the day agent, so a scan briefly slows day ticks; the broader sp500 scan
    # takes longer, so a 15-min cadence keeps that contention infrequent while
    # still keeping ideas fresh. Price staleness for sizing is separately handled
    # by _reprice_to_market. Set to 0 to disable the refresh job.
    scan_refresh_seconds: int = 900
    scan_refresh_timeout_seconds: float = 420.0  # client wait cap (sp500 ~500 names)
    # Which scanner universe to scan. "liquid" (~300 hardcoded large-caps, semis/
    # tech-tilted) can leave the agent picking from a narrow, correlated set on a
    # given day. "sp500" (~500 dynamically-maintained S&P 500 names across every
    # sector) gives a much broader, more diversified candidate pool so the agent
    # isn't stuck on a handful of names. Options: liquid | sp500 | all.
    scan_universe: str = "sp500"
    # Intraday stop monitor. The day agent sets a stop at entry but nothing
    # submits it as a live order, so a dedicated cheap job (no LLM) checks each
    # open position's live price against its stop/target on this cadence and
    # closes breached positions — catching fast moves between the slower LLM
    # ticks. Set to 0 to disable.
    stop_monitor_seconds: int = 30
    # Alpaca leg is the source of truth — retry transient order failures
    # (spurious trading_disabled, transport errors) before giving up.
    alpaca_max_retries: int = 4
    alpaca_retry_backoff: float = 1.0   # seconds, base for exponential backoff
    alpaca_retry_cap: float = 6.0       # max seconds per retry wait
    # Overall wall-clock budget for one place_order retry sequence; caps how long
    # a slow MCP restart + retries can block the scheduler tick.
    alpaca_place_budget_seconds: float = 25.0

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
    # Mandatory diligence: reject any new trade whose symbol did not receive BOTH
    # a get_news and a get_analyst_view call this tick. The checklist marks these
    # "required"; without enforcement the model skips them under time pressure and
    # the low-diligence proposals are exactly the low-quality trades. Set False to
    # revert to advisory-only.
    day_require_diligence: bool = True
    # Compact mode: send a much smaller request (short system prompt, trimmed idea
    # list, fewer tool-loop steps) so the payload fits under the fallback model's
    # ~8k-token input cap when the primary (gpt-5) is rate-limited. This is the
    # DEFAULT; the live `compact_prompt` setting (a UI toggle) overrides it.
    day_compact_mode: bool = False
    # Catch-all protective stop for any HELD position that has no explicit plan
    # (e.g. opened manually, or before the plan feature existed). The stop monitor
    # backfills a stop this % from avg cost (and a 2R target) so every position is
    # protected, not just freshly-proposed ones. Default 0 = OFF (don't fabricate
    # stops on positions whose intended stop we don't know). Set e.g. 0.05 for a
    # 5% safety-net stop.
    day_default_stop_pct: float = 0.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Merge secrets.toml into os.environ (env wins if already set).
    for k, v in _load_secrets_file().items():
        os.environ.setdefault(k, v)
    return Settings()  # type: ignore[call-arg]


def db_path() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / "sandbox.sqlite"
