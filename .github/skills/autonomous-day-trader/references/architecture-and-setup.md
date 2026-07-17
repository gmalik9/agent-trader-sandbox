# Architecture & Setup

## The three repos

```
short-term-trader  ──(MCP: ideas, quotes, 9-source news)──┐
                                                          ▼
stock-recommender  ──(MCP: Alpaca paper broker + analyst)──►  agentic-trader
                                                              (DayTraderAgent
                                                               + scheduler + UI)
                                                                     │
                                                                     ▼
                                                          Alpaca PAPER account
```

- **agentic-trader** (this repo, `gmalik9/agent-trader-sandbox`) — the trader.
  - `DayTraderAgent` (`src/agents/day_trader.py`): the intraday decision loop.
  - `scheduler` (`src/scheduler/runner.py`): ticks `day_tick` (~2 min), a
    `stop_monitor` (~30 s), `scan_refresh` (~15 min), `mtm`/`reconcile` (1 min),
    plus a `day-tick watchdog` that self-heals a hung tick.
  - `DualBroker`: **Alpaca paper = primary (source of truth)**, local sandbox =
    mirror. `src/brokers/`.
  - Storage: `data/sandbox.sqlite` (WAL) + `data/reasoning_log.jsonl`.
  - UI: Streamlit dashboard on `http://localhost:${WEB_PORT:-8502}`.
- **short-term-trader** (`gmalik9/short-term-stock-recommender`) — read-only
  intraday scanner. MCP tools: `market_status`, `scan_latest`, `scan_run`,
  `list_ideas`, `lookup_ticker`, `get_news`. ~15-min-delayed yfinance data +
  Finnhub + 9-source VADER news sentiment. Universes: `liquid` (~300),
  `sp500` (~500, default here), `all` (~6000).
- **stock-recommender** (`gmalik9/long-term-stock-recommender`) — Alpaca paper
  broker MCP (account, positions, `place_order`, `close_position`) + analyst
  views. Hard-coded to `https://paper-api.alpaca.markets`.

## How to run

```bash
# from the agentic-trader repo root
./trader start            # start scheduler + dashboard (detached)
./trader rebuild          # rebuild images (REQUIRED after editing .py/app.py)
./trader status           # container status
./trader logs scheduler   # follow scheduler logs
./trader tick day         # queue one manual day tick
./trader stop             # stop containers (data/ ledger preserved)
```

`restart` alone does **not** reload code — `app.py`/`src/` are baked into the
image via `COPY . .`. Always `./trader rebuild` (or `docker compose up -d
--build <svc>`) after code edits.

Two services: `scheduler` (owns all ticking) and `web` (Streamlit UI). The
sibling repos are mounted read-write and launched as MCP subprocesses by the
scheduler.

## Environment variables

Non-secret flags are pinned in `docker-compose.yml`; **secrets live in
`.streamlit/secrets.toml`** (git-ignored). The skill is permitted to read the
news-source and Alpaca keys from there / the container env to do its job — but
never print or commit them.

| Var | Required when | Purpose |
|---|---|---|
| `LLM_PROVIDER` | always (default `github`) | `github` \| `openai` \| `anthropic` |
| `LLM_MODEL` | optional | e.g. `openai/gpt-5` (primary; downshifts to `gpt-4o-mini`) |
| `GITHUB_TOKEN` | if `LLM_PROVIDER=github` | PAT with GitHub Models access (free tier rate-limits) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | if that provider | removes the free-tier caps — the durable fix for throttling |
| `BROKER_BACKEND` | always (default `dual`) | `sandbox` \| `alpaca_paper` \| `dual` |
| `ALPACA_API_KEY_ID` / `ALPACA_SECRET_KEY` | broker ∈ {alpaca_paper, dual} | Alpaca **paper** creds |
| `ALPACA_PAPER` | same; **must be `true`** | safety assertion (paper only) |
| `STOCK_REC_MCP_TRADING_ENABLED` | same; must be `true` | unlocks upstream write tools |
| `SHORT_TERM_TRADER_PATH` | always | path to short-term-trader repo (in-container `/sibling/...`) |
| `STOCK_RECOMMENDER_PATH` | always | path to stock-recommender repo |
| `FINNHUB_API_KEY` | optional | news / earnings; passed to MCP subprocesses |
| `ALPHAVANTAGE_API_KEY` | optional | news; passed to MCP subprocesses |
| `SCAN_UNIVERSE` | optional (default `sp500`) | `liquid` \| `sp500` \| `all` |
| `DAY_AUTOPILOT` | optional (default `true`) | `false` = Copilot-driven (no scheduler LLM/PAT calls); live `day_autopilot` setting overrides |
| `DAY_COMPACT_MODE` | optional (default `false`) | default for compact requests (live `compact_prompt` setting overrides) |
| `CAPITAL_TOTAL` / `SPLIT_DAY_PCT` | optional | bankroll + day/long split |

## Copilot-driven vs scheduler-autopilot (`day_autopilot`)

Live toggle (no rebuild). Stops, scan refresh, reconcile, MTM and the end-of-day
force-flat run in **both** modes. The two never both trade — pick one:

- **`off` — Copilot-driven (this skill).** `job_day_tick` short-circuits (no LLM
  call). Copilot is the brain: read `gather_context.py`, reason, act via
  `execute_trade.py`. No PAT, no rate-limit ceiling.
- **`on` — the standalone scheduler agent.** `job_day_tick` calls the configured
  LLM (`LLM_PROVIDER`) every `DAY_TICK_SECONDS` and trades on its own — the
  classic engine, unchanged. Uses the PAT; bounded by that provider's limits.

Set it:
```bash
docker compose exec -T scheduler python -c \
 "from src.sandbox import db as d; from src.config import db_path; \
  d.set_setting(d.get_conn(db_path()),'day_autopilot','off')"   # or 'on'
```

News-source keys the skill may use (all optional, all in `secrets.toml` / env,
consumed by the scanner MCP): `FINNHUB_API_KEY`, `ALPHAVANTAGE_API_KEY`, plus any
Marketaux/NewsAPI/Tiingo/Reddit keys the sibling scanner reads. The scanner
already aggregates 9 sources into a single `sentiment_score`.

## Health checks

```bash
# containers up?
./trader status

# is the agent ticking / throttled? (read-only)
docker compose exec -T scheduler python .github/skills/autonomous-day-trader/scripts/session_snapshot.py

# market open? live Alpaca account + positions
docker compose exec -T scheduler python -c "
from src.mcp_clients.long_term import LongTermClient
lt=LongTermClient(); lt.start()
try:
    print(lt.get_account())
    for p in lt.list_positions(): print(p['symbol'], p.get('qty'), p.get('unrealized_pl'))
finally: lt.stop()
"
```

## Safety guarantees (from AGENTS.md)

- No live-broker code path. Alpaca leg asserts `ALPACA_PAPER=true` and that the
  account number starts with `PA` before any write.
- Kill-switch: `settings(key='kill_switch', value='on')` halts every agent within
  one tick (≤ 5 s).
- Caps enforced in-code and via the upstream MCP.
- Every order/fill/cash-move/agent-run is audited in `data/sandbox.sqlite`; a
  `reconcile` job syncs Alpaca statuses every minute.

## Known operational quirks

- **Deploy = rebuild** (not restart) — code is baked into the image.
- **SQLite is WAL**: inspect read-only (`file:...?mode=ro`); never add concurrent
  writers (corrupted the DB once).
- **Throttling**: GitHub Models free tier 429s often; the fallback `gpt-4o-mini`
  has an ~8k-token input cap → 413 on the full prompt. Enable **compact mode**
  to keep trading; a paid OpenAI/Anthropic key is the real fix.
- **Alpaca refuses to short many leveraged/inverse ETFs** (422) — the agent buys
  the inverse counterpart instead (e.g. short SOXL → long SOXS).
- **Options** are managed on the options venue and expire on their own; they
  can't be closed via the stock `close_position` endpoint.
