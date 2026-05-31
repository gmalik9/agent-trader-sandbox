# Plan — Agentic Trader & Paper Sandbox

**Repo:** https://github.com/gmalik9/agent-trader-sandbox.git
**Local path:** `/Users/girikmalik/Documents/girik_academic_resources/personal_projects/agentic-trader/`

## Goal

Build a Python project that runs two LLM-driven trading agents — a **Day-Trader**
(consumes the `short-term-trader` repo via MCP) and a **Long-Term Investor**
(consumes the `stock-recommender` repo via MCP) — against an **in-house
paper-trading sandbox** with a pluggable Alpaca-paper backend. A **Coordinator**
splits a configurable bankroll between the two sub-accounts. A **Streamlit
dashboard** shows the equity curve, trade blotter, positions, PnL and each
agent's reasoning trace. An APScheduler-driven loop runs the agents on a
schedule, and a "Tick now" button forces an out-of-band run.

## Non-goals

- Live real-money trading. The sandbox refuses anything but paper/simulated routes.
- Options, crypto, futures, FX. Equities + ETFs only (matches both upstream repos).
- Historical backtesting engine. Reserved for a future replay-mode clock.
- Multi-broker routing beyond `sandbox` and `alpaca_paper`.

## Inputs we already have

- `short-term-trader/` — intraday scanners, tier-A/B/C trade ideas with
  entry / 1.5×ATR stop / 2R target, dual-cadence scan orchestrator, MCP server
  on stdio, REST on `:8503`, Streamlit on `:8502`, SQLite audit log.
  See `/memories/repo/short-term-trader-integration-guide.md`.
- `stock-recommender/` — long-term screener + $5k inverse-beta allocator,
  multi-source news + VADER sentiment, 12-tool MCP server (7 READ / 5 WRITE),
  Alpaca paper wrapper, SQLite audit log, hard sandbox guards.
- A GitHub PAT tied to the user's Copilot account (for GitHub Models API).

## High-level approach

1. **Sandbox first.** Own ledger in SQLite, deterministic fill engine,
   `BrokerBase` interface. Alpaca paper is a second `BrokerBase` impl that
   delegates to the upstream `stock-recommender` MCP — we never re-implement
   Alpaca auth or safety here. A third impl, `DualBroker`, fans every order
   out to **both** sandbox and Alpaca paper in parallel so the user can
   compare agent performance side-by-side against an independent venue.
2. **Talk to upstream over MCP stdio**, using the official `mcp` Python SDK
   with `ClientSession` + `stdio_client`. Typed Python wrappers per repo.
3. **LLM via GitHub Models API** (`https://models.github.ai/inference`) using a
   `GITHUB_TOKEN` PAT; OpenAI and Anthropic providers behind the same protocol
   as fallbacks. Function-calling loop with bounded `max_steps`.
4. **Two specialist agents + Coordinator.** Each agent owns a sub-account.
   Hard risk rules in code; LLM decides which ideas to take and writes a
   thesis per trade.
5. **Scheduler in its own process**, Streamlit is read-only over SQLite so the
   UI never blocks on agent work.

## Phases

### Phase 1 — Scaffolding & repo hygiene
- Create the layout (see `design.md` §"Repository layout").
- Pin deps in `requirements.txt`: `streamlit`, `pandas`, `numpy`, `plotly`,
  `mcp`, `apscheduler`, `pydantic`, `httpx`, `pytest`, `pytest-asyncio`,
  `python-dotenv`, `yfinance`, `pandas_market_calendars`, `responses`.
- `.streamlit/secrets.toml.example` documenting every env var.
- `run.sh`, `Dockerfile`, `docker-compose.yml`, `.gitignore`, `.dockerignore`.
- Initialize git, set remote to `git@github.com:gmalik9/agent-trader-sandbox.git`.

### Phase 2 — Sandbox paper broker  *(parallel with Phase 3)*
- SQLite DDL in `src/sandbox/schema.sql` (tables in `design.md` §"Data model").
- `src/sandbox/clock.py` — real-time clock with US-market-hours helper via
  `pandas_market_calendars`.
- `src/sandbox/engine.py` — fill model: market = next-bar close + slippage bps;
  limit = filled iff bar `[low, high]` touches the limit; configurable
  commission bps; fractional shares allowed; cash-conservation invariant
  asserted on every order.
- `src/brokers/base.py` — `BrokerBase` interface
  (`get_account`, `list_positions`, `place_order`, `cancel_order`,
  `close_position`, `mark_to_market`, `equity_curve`).
- `src/brokers/sandbox_broker.py` — sandbox impl.
- `src/brokers/alpaca_paper_broker.py` — delegates to `stock-recommender` MCP
  WRITE tools; mirrors all writes into our `orders` table as `routed_external`.
- `src/brokers/dual_broker.py` — composite `BrokerBase` that wraps a
  `primary` (sandbox) and a `secondary` (alpaca_paper). Every write
  (`place_order`, `cancel_order`, `close_position`) is issued to both; reads
  (`get_account`, `list_positions`, `equity_curve`) come from the primary,
  while the secondary's snapshots are stored under a parallel account name
  (`day_alpaca`, `long_alpaca`) so the UI can render both equity curves
  side-by-side. Order rows are linked via a shared `dual_group_id` so per-
  trade slippage between the two venues is queryable.
- Tests: fill determinism, cash conservation, short-sale rejection, blocklist,
  per-order USD cap, MTM correctness on multi-leg position, **DualBroker
  fan-out** (one decision → two `orders` rows with the same `dual_group_id`,
  one secondary failure does not roll back the primary, divergence is
  recorded not raised).

### Phase 3 — MCP clients to upstream projects  *(parallel with Phase 2)*
- `src/mcp_clients/base.py` — subprocess lifecycle, auto-restart, JSON-RPC
  request/response with timeouts.
- `src/mcp_clients/short_term.py` — typed wrappers around the
  `short-term-trader` MCP tools (`get_intraday_ideas`, `run_scan`,
  `get_ticker_quote`, etc.).
- `src/mcp_clients/long_term.py` — typed wrappers around the
  `stock-recommender` MCP tools (`get_recommendations`,
  `get_portfolio_suggestion`, `lookup_ticker`, `get_news`,
  `get_account`, `list_positions`, `place_order`, …).
- Contract tests pin expected tool names and JSON schemas; failure = upstream
  drift, fix immediately.
- If we need a tool the upstream doesn't expose, add it as a new **READ-only**
  tool in that upstream repo and record the diff in
  `UPSTREAM_PATCHES.md`. We do not modify upstream WRITE/safety logic.

### Phase 4 — LLM layer (GitHub Models first)
- `src/llm/provider.py` — `LLMProvider` protocol:
  `chat(messages, tools=None, **kw) -> ChatResult`.
- `src/llm/github_models.py` — POST `https://models.github.ai/inference/chat/completions`
  with `Authorization: Bearer ${GITHUB_TOKEN}`. Default model
  `openai/gpt-4o-mini`, configurable via `LLM_MODEL`.
- `src/llm/openai_provider.py`, `src/llm/anthropic_provider.py` — fallbacks.
- `src/llm/tool_loop.py` — generic function-calling loop: model → tool call →
  tool result → model, capped at `max_steps` (default 8); every step appended
  to `agent_runs.tools_called` as JSON.
- `src/llm/factory.py` — picks provider from `LLM_PROVIDER`; fails loud with an
  actionable message if no key is set.
- **Note on Copilot SDK.** There is no public Copilot SDK for arbitrary Python
  apps. GitHub Models is the closest GitHub-account-auth path. True
  Copilot-subscription billing is only reachable from a VS Code extension via
  the VS Code Language Model API — out of scope here, can be added later as
  another `LLMProvider` impl living inside a thin extension.

### Phase 5 — Agents
- `src/agents/base.py` — `Agent` ABC with `name`, `broker`, `llm`, `mcp_client`,
  `policy`, `run_once(now) -> list[Decision]`. Every `run_once` writes one
  `agent_runs` row capturing prompt, response, tool calls, decisions.
- `src/agents/day_trader.py`
  - Pulls fresh tier-A intraday ideas from `short_term` MCP, filters by
    `heat_score` threshold and signal-tag whitelist.
  - Uses upstream's `entry / stop / target`; sizes 1% account-risk per trade
    against the ATR-derived stop already provided.
  - Hard rules in code: max N concurrent positions, no overnight holds (flat
    by `15:55 ET`), daily-drawdown halt at `-2%`.
  - LLM role: pick *which* ideas to take, optional skip-day if regime is
    choppy, write one-paragraph thesis per trade.
- `src/agents/long_term.py`
  - Calls `stock-recommender` MCP daily for `get_recommendations` and
    `get_portfolio_suggestion(budget=current_equity)`.
  - Diffs vs current positions, throttled to one rebalance per week unless
    drift > 10%.
  - LLM role: sanity-check picks against the latest news from `get_news`,
    optionally veto names, write portfolio commentary.
- `src/agents/coordinator.py` — owns the master bankroll account split into
  two sub-accounts (`day` 30%, `long` 70% by default). On the first trading
  day of each month, rebalances cash between sub-accounts back toward target
  weights via the `cash_ledger`.
- `src/agents/policy.py` — shared risk gates: max position % of sub-account,
  blocklist (inherits upstream blocklist), kill-switch flag read from DB.

### Phase 6 — Scheduler & run loop
- `src/scheduler/runner.py` — APScheduler `BlockingScheduler`:
  - Day-trader: every 5 min during US market hours.
  - Long-term: daily at `16:30 ET`.
  - Coordinator: first trading day of the month at `16:45 ET`.
  - Mark-to-market: every 1 min during market hours; appends to `equity_curve`.
- File-lock singleton in `data/` so the scheduler can't double-start.
- Manual `tick(agent_name)` callable from the Streamlit UI by writing a row
  into a `tick_requests` table that the scheduler polls every 5 s.

### Phase 7 — Streamlit dashboard
- `app.py` tabs:
  - **Overview** — total + per-agent equity curves (Plotly), today's PnL, cash,
    positions value, kill-switch toggle.
  - **Day-Trader** — blotter, open positions w/ unrealized PnL, latest
    reasoning trace, top current intraday ideas pulled live from MCP.
  - **Long-Term Investor** — current allocation donut, target-vs-actual drift,
    last rebalance, latest recommendations + news summary.
  - **History** — full trade log filterable by agent/symbol/date, CSV export.
  - **Agent Runs** — chronological reasoning trace with prompt / response /
    tool-call expanders.
  - **Settings** — backend switch (`sandbox` ↔ `alpaca_paper`), LLM provider,
    capital + split, "Tick now" buttons per agent.
- UI is **read-only** over SQLite; all writes happen in the scheduler process.

### Phase 8 — Verification, docs, packaging
- `tests/` — unit (sandbox engine, brokers, policy gates), integration
  (full tick with mocked LLM + mocked MCP), contract (upstream MCP tool
  schemas).
- `README.md`, `AGENTS.md` (sandbox safety guarantees, env vars, kill-switch),
  refreshed `design.md` and `tasks.md`.
- `Dockerfile` + `docker-compose.yml` with bind-mounts to the two sibling
  repos so the MCP subprocesses can launch in-container.
- `run.sh` for local one-command start (scheduler in background, Streamlit in
  foreground).

## Verification

1. `pytest -q` green for all unit + integration + contract tests.
2. `bash run.sh`, open Streamlit, hit "Tick day-trader" with sandbox backend →
   new orders + fills in blotter, `equity_curve` updates, reasoning visible.
3. "Tick long-term" → rebalance plan generated against current long sub-account
   equity; orders fill against yfinance prices.
4. Set `BROKER_BACKEND=dual`, re-tick → for each agent decision there are
   exactly two `orders` rows sharing a `dual_group_id` (one sandbox, one
   `routed_external`); the order also appears in the Alpaca paper account;
   the Overview tab renders both equity curves on the same axes.
5. Toggle kill-switch in UI → next scheduled tick exits early with `halted`
   status row in `agent_runs`.
6. Leave scheduler running for a full market session → minute-level equity
   curve, no duplicate orders, MCP subprocess auto-restart works after SIGKILL.
7. Set `LLM_PROVIDER=github` + `GITHUB_TOKEN` → a real completion is persisted
   in `agent_runs.response`.

## Decisions

- **Sandbox + Alpaca + Dual** pluggable via `BrokerBase`. `BROKER_BACKEND`
  takes `sandbox` | `alpaca_paper` | `dual`. **Dual is the recommended
  setting** because it makes the local-vs-Alpaca comparison the whole point
  of running the agent.
- **LLM = GitHub Models API via PAT**, with OpenAI/Anthropic fallback.
- **Two agents + Coordinator**, sub-accounts kept separate in the ledger so
  per-agent PnL is trivial.
- **Both scheduler + manual tick.** Scheduler runs as a **separate process**
  from Streamlit.
- **Capital: $100k, 30% day-trader / 70% long-term**, editable in Settings.
- **Consume upstream via MCP stdio.** Allowed to add new READ tools upstream
  if missing; never modify upstream WRITE/safety logic.
- **Fill-price source v1:** yfinance bars (delayed ~15 min); accept the delay
  and timestamp fills accordingly.
- **Reasoning-trace storage v1:** full text in SQLite.
- **Process topology v1:** separate scheduler process, Streamlit read-only.

## Open follow-ups

1. Revisit fill-price source once we have data on slippage vs delayed-price drift.
2. Optional `replay` mode for `sandbox/clock.py` to enable historical backtests.
3. Optional VS Code extension wrapper for true Copilot-subscription LLM access.
