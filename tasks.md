# Tasks — Agentic Trader & Paper Sandbox

**Repo:** https://github.com/gmalik9/agent-trader-sandbox.git

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done.

Each phase has an **acceptance test** that must pass before the phase is
considered done.

---

## Phase 1 — Scaffolding & repo hygiene

- [ ] 1.1  Create directory tree from `design.md` §2.
- [ ] 1.2  Write `requirements.txt` with pinned deps (streamlit, pandas,
            numpy, plotly, mcp, apscheduler, pydantic, pydantic-settings,
            httpx, pytest, pytest-asyncio, python-dotenv, yfinance,
            pandas_market_calendars, responses).
- [ ] 1.3  Write `runtime.txt` (`python-3.11`).
- [ ] 1.4  Write `.gitignore` (data/, .venv/, __pycache__/, .pytest_cache/,
            .streamlit/secrets.toml, *.sqlite, *.lock).
- [ ] 1.5  Write `.dockerignore`.
- [ ] 1.6  Write `.streamlit/secrets.toml.example` documenting every env var
            from `design.md` §10.
- [ ] 1.7  Write `Dockerfile` (python:3.11-slim, install deps, copy src,
            entrypoint = `bash run.sh`).
- [ ] 1.8  Write `docker-compose.yml` with `scheduler` + `web` services
            sharing the `data/` volume and bind-mounting the two sibling
            repos read-only.
- [ ] 1.9  Write `run.sh` (start `python -m src.scheduler.runner &` then
            `streamlit run app.py`).
- [ ] 1.10 Initialize git, create `main` branch, set remote
            `git@github.com:gmalik9/agent-trader-sandbox.git`. **Do not push
            until Phase 2 is green.**
- [ ] 1.11 Write skeleton `README.md` and `AGENTS.md` (sandbox safety
            statement + env vars table).

**Acceptance:** `pip install -r requirements.txt` succeeds in a fresh venv;
`docker compose build` succeeds.

---

## Phase 2 — Sandbox paper broker  *(parallel with Phase 3)*

- [ ] 2.1  Write `src/sandbox/schema.sql` (all tables from `design.md` §3).
- [ ] 2.2  Write `src/sandbox/db.py` — `get_conn()`, `migrate()`, idempotent.
- [ ] 2.3  Write `src/sandbox/clock.py` — `now_utc()`,
            `is_market_open(ts)`, `next_open(ts)`, `next_close(ts)` using
            `pandas_market_calendars` (`XNYS`).
- [ ] 2.4  Write `src/sandbox/engine.py`:
  - [ ] 2.4.1 `get_bar(symbol, ts)` → 1-min yfinance bar, on-disk cache
              keyed by `(symbol, date)`.
  - [ ] 2.4.2 `simulate_fill(order, bar, *, slippage_bps, commission_bps)` →
              `FillResult`.
  - [ ] 2.4.3 `mark_to_market(account_id)` → write rows to
              `positions_snapshot` and `equity_curve`.
  - [ ] 2.4.4 Assert cash conservation after every fill.
- [ ] 2.5  Write `src/brokers/base.py` — dataclasses + `BrokerBase` ABC.
- [ ] 2.6  Write `src/brokers/sandbox_broker.py` implementing `BrokerBase`,
            applying blocklist + per-order USD cap + per-symbol % cap.
- [ ] 2.7  Write `src/brokers/alpaca_paper_broker.py` delegating to
            `mcp_clients/long_term.py` WRITE tools; mirror every order into
            local `orders` with `status='routed_external'` and poll for
            `external_id` resolution.
- [ ] 2.8  Write `src/brokers/dual_broker.py` — fan-out wrapper:
            mints `dual_group_id`, submits to both legs in parallel via
            `ThreadPoolExecutor`, routes sub-accounts (`day`→`day`+`day_alpaca`,
            `long`→`long`+`long_alpaca`), records `dual_divergence` rows on
            secondary failure or fill mismatch, never lets a secondary error
            roll back the primary.
- [ ] 2.9  Tests:
  - [ ] 2.9.1 `tests/test_sandbox_engine.py` — fill determinism, limit fill
              iff in `[low, high]`, slippage applied, commission deducted.
  - [ ] 2.9.2 `tests/test_sandbox_broker.py` — cash conservation,
              short-sale rejection, blocklist, per-order cap, per-symbol cap.
  - [ ] 2.9.3 `tests/test_alpaca_broker.py` — with a mocked MCP client,
              assert `place_order` writes both rows correctly.
  - [ ] 2.9.4 `tests/test_dual_broker.py` — one decision → two `orders`
              rows sharing `dual_group_id`; secondary exception does not
              raise and writes a `dual_divergence` row; cancel/close
              cancels both legs; reads default to primary, `venue='alpaca_paper'`
              kwarg returns secondary view.

**Acceptance:** `pytest tests/test_sandbox_engine.py tests/test_sandbox_broker.py
tests/test_alpaca_broker.py tests/test_dual_broker.py -q` green.

---

## Phase 3 — MCP clients  *(parallel with Phase 2)*

- [ ] 3.1  Write `src/mcp_clients/base.py` — `MCPClient` with `start`,
            `stop`, `restart`, `call(tool, **args)`, `health()`. Uses
            `mcp.client.stdio.stdio_client` + `ClientSession`. Auto-restart
            after 3 consecutive failures.
- [ ] 3.2  Write `src/mcp_clients/short_term.py` — typed wrappers for the
            tools we need (`get_intraday_ideas`, `run_scan`,
            `get_ticker_quote`; extend as Phase 5 surfaces more).
- [ ] 3.3  Write `src/mcp_clients/long_term.py` — typed wrappers for
            `get_recommendations`, `get_portfolio_suggestion`,
            `lookup_ticker`, `get_news`, `get_account`, `list_positions`,
            `list_orders`, `place_order`, `cancel_order`,
            `cancel_all_orders`, `close_position`,
            `rebalance_to_recommendations`.
- [ ] 3.4  Contract tests (`tests/test_mcp_clients.py`) — skipped when the
            sibling-repo path env vars are unset; otherwise spawn the real
            servers, assert `tools/list` contains every name we depend on
            with matching argument schemas.
- [ ] 3.5  If any required tool is missing upstream, add it as a READ-only
            tool there and record the diff in `UPSTREAM_PATCHES.md`.

**Acceptance:** `pytest tests/test_mcp_clients.py -q` green (or all skipped
with a clear reason); manual `python -c "from src.mcp_clients.short_term
import ShortTermClient; c=ShortTermClient(); c.start();
print(c.get_intraday_ideas(tier='A')[:2])"` returns rows.

---

## Phase 4 — LLM layer

- [ ] 4.1  Write `src/llm/provider.py` (protocol + dataclasses).
- [ ] 4.2  Write `src/llm/github_models.py` (httpx client, OpenAI-style
            function calling, retries on 429/5xx).
- [ ] 4.3  Write `src/llm/openai_provider.py` and
            `src/llm/anthropic_provider.py`.
- [ ] 4.4  Write `src/llm/tool_loop.py` (capped loop, structured trace).
- [ ] 4.5  Write `src/llm/factory.py` (resolve provider + model from
            settings then env).
- [ ] 4.6  Tests:
  - [ ] 4.6.1 `tests/test_llm_tool_loop.py` — fake provider that scripts
              `[tool_call, tool_call, text]`; assert max-step cap, trace
              capture, error propagation.
  - [ ] 4.6.2 `test_github_models.py` — `responses`-mocked HTTP, asserts
              `Authorization` header and request body shape.

**Acceptance:** `pytest tests/test_llm_tool_loop.py
tests/test_github_models.py -q` green; ad-hoc smoke
`python -m src.llm.factory --smoke` with a real `GITHUB_TOKEN` returns text.

---

## Phase 5 — Agents

- [ ] 5.1  Write `src/agents/policy.py` — `validate(decision, broker)` +
            kill-switch read.
- [ ] 5.2  Write `src/agents/base.py` — `Agent` ABC + `agent_runs` writer.
- [ ] 5.3  Write `src/agents/day_trader.py` — hard rules in code, LLM
            picks ideas, ATR-based sizing, force-flat at 15:55 ET, daily
            drawdown halt.
- [ ] 5.4  Write `src/agents/long_term.py` — weekly throttle, 10%-drift
            override, LLM sanity-check against news, propose rebalance.
- [ ] 5.5  Write `src/agents/coordinator.py` — monthly cash transfer
            between sub-accounts.
- [ ] 5.6  Tests:
  - [ ] 5.6.1 `tests/test_policy.py`.
  - [ ] 5.6.2 `tests/test_day_trader.py` — fake MCP + fake LLM, assert
              orders match scripted decisions, kill-switch halts, drawdown
              halt triggers.
  - [ ] 5.6.3 `tests/test_long_term.py` — throttle respected; drift override
              triggers; rebalance plan correct vs target weights.
  - [ ] 5.6.4 `tests/test_coordinator.py` — transfer math, refuses on
              kill-switch.

**Acceptance:** `pytest tests/test_policy.py tests/test_day_trader.py
tests/test_long_term.py tests/test_coordinator.py -q` green.

---

## Phase 6 — Scheduler & run loop

- [ ] 6.1  Write `src/scheduler/runner.py` — APScheduler `BlockingScheduler`,
            jobs per `design.md` §8, `tick_requests` poller, file-lock
            singleton.
- [ ] 6.2  Wire `settings`-table re-read at the start of every job.
- [ ] 6.3  Tests (`tests/test_scheduler.py`) — fake clock, assert jobs fire
            at the expected wall times; `tick_requests` row drains and runs
            the right agent.

**Acceptance:** `python -m src.scheduler.runner --once day` prints a
day-trader run summary; `tests/test_scheduler.py` green.

---

## Phase 7 — Streamlit dashboard

- [ ] 7.1  Write `app.py` with six tabs from `plan.md` §Phase 7.
- [ ] 7.2  Overview tab: total + per-agent equity curves (Plotly) with the
            sandbox and Alpaca legs plotted on shared axes when
            `BROKER_BACKEND=dual`; today's PnL, cash, positions value,
            kill-switch toggle; a small "divergence" KPI summarizing
            `dual_divergence` rows in the last 24h.
- [ ] 7.3  Day-Trader tab: blotter, open positions w/ unrealized PnL,
            latest reasoning trace, live tier-A ideas from MCP.
- [ ] 7.4  Long-Term tab: allocation donut, drift table, last-rebalance
            timestamp, recommendations + news summary.
- [ ] 7.5  History tab: filterable trade log + CSV export; when
            `BROKER_BACKEND=dual`, rows of the same `dual_group_id` are
            grouped together showing the sandbox-vs-Alpaca fill diff.
- [ ] 7.6  Agent Runs tab: expanders per run showing prompt, response,
            tools_called JSON.
- [ ] 7.7  Settings tab: backend switch, LLM provider/model, capital +
            split, "Tick now" buttons per agent.
- [ ] 7.8  Smoke: launch `bash run.sh`, click through each tab.

**Acceptance:** Smoke checklist below all pass.

---

## Phase 8 — Verification, docs, packaging

- [ ] 8.1  Flesh out `README.md` — quickstart, env vars, screenshots,
            safety statement, troubleshooting.
- [ ] 8.2  Flesh out `AGENTS.md` — explicit sandbox safety guarantees, env
            vars, kill-switch, blocklist, caps, MCP tool list per agent.
- [ ] 8.3  `Dockerfile` + `docker-compose.yml` final pass.
- [ ] 8.4  Run full `pytest -q` — must be green.
- [ ] 8.5  Smoke acceptance:
  - [ ] 8.5.1 `bash run.sh` → Streamlit reachable on `:8501`.
  - [ ] 8.5.2 "Tick day-trader" with sandbox backend → orders/fills in
              blotter, equity_curve updates, reasoning visible.
  - [ ] 8.5.3 "Tick long-term" → rebalance plan; orders fill at yfinance
              prices.
  - [ ] 8.5.4 `BROKER_BACKEND=dual`, re-tick → each decision produces two
              `orders` rows with the same `dual_group_id` (one sandbox, one
              `routed_external`); the order also appears in the Alpaca
              paper account; Overview renders both equity curves on shared
              axes; History groups the two legs together.
  - [ ] 8.5.5 Kill-switch on → next tick row has `status='halted'`.
  - [ ] 8.5.6 One full market session: minute-level equity curve, no dup
              orders, MCP subprocess auto-restart works after `kill -9`.
  - [ ] 8.5.7 `LLM_PROVIDER=github` + `GITHUB_TOKEN` → real completion text
              persisted in `agent_runs.response`.
- [ ] 8.6  First `git push origin main` to
            `git@github.com:gmalik9/agent-trader-sandbox.git`.

**Acceptance:** every smoke item above ticked.

---

## Cross-cutting tasks (do whenever encountered)

- [ ] X.1  Keep `UPSTREAM_PATCHES.md` up to date whenever we touch the
            sibling repos.
- [ ] X.2  Update `/memories/repo/short-term-trader-integration-guide.md` and
            the session report whenever upstream tool schemas change.
- [ ] X.3  Never commit `.streamlit/secrets.toml` or anything under `data/`.
- [ ] X.4  Never bypass `policy.validate` or `kill_switch`; never widen the
            blocklist without an explicit user request.
