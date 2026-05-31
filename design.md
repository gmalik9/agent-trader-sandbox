# Design — Agentic Trader & Paper Sandbox

**Repo:** https://github.com/gmalik9/agent-trader-sandbox.git

## 1. System overview

```
                +-------------------------+
                |   Streamlit dashboard   |  read-only
                |   (app.py, port 8501)   |<-----------+
                +-------------------------+            |
                                                       |
                                                  SQLite
                                              data/sandbox.sqlite
                                                       ^
                                                       | writes
                                                       |
+-----------------+      +--------------------------+  |
|  APScheduler    |----->|   Agents (in-process)    |--+
|  (own process)  |      |                          |
|  runner.py      |      |  Coordinator             |
+-----------------+      |  ├─ DayTraderAgent       |
                         |  └─ LongTermAgent        |
                         +-----+----------+---------+
                               |          |
                       BrokerBase     LLMProvider
                       (pluggable)    (pluggable)
                          /    \           |
                         /      \          v
                        v        v   GitHub Models / OpenAI / Anthropic
                   DualBroker (default)
                     /        \
                    v          v
            SandboxBroker  AlpacaPaperBroker
                |                |
                v                v
        SQLite ledger    short-term-trader MCP
                         stock-recommender MCP   (Alpaca paper inside)
                                |
                                v
                       MCP stdio subprocesses
                       managed by mcp_clients/
```

Two **processes** at runtime:

1. `scheduler` — APScheduler + agents + brokers + LLM + MCP subprocess clients.
   Writes everything to SQLite.
2. `streamlit` — pure reader over SQLite; the only writes it performs are
   `tick_requests` rows and `settings` updates that the scheduler picks up.

## 2. Repository layout

```
agentic-trader/
├── app.py                       # Streamlit UI (read-only over SQLite)
├── run.sh                       # starts scheduler + streamlit
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── runtime.txt
├── .streamlit/
│   └── secrets.toml.example
├── .gitignore
├── .dockerignore
├── README.md
├── AGENTS.md
├── design.md
├── plan.md
├── tasks.md
├── UPSTREAM_PATCHES.md          # any diffs we land in the two sibling repos
├── data/                        # gitignored; SQLite + file locks live here
├── src/
│   ├── __init__.py
│   ├── config.py                # env + secrets loader (pydantic Settings)
│   ├── sandbox/
│   │   ├── __init__.py
│   │   ├── schema.sql
│   │   ├── db.py                # connection + migrations
│   │   ├── clock.py             # market-hours helpers
│   │   └── engine.py            # fill simulator + MTM
│   ├── brokers/
│   │   ├── __init__.py
│   │   ├── base.py              # BrokerBase ABC + dataclasses
│   │   ├── sandbox_broker.py
│   │   ├── alpaca_paper_broker.py
│   │   └── dual_broker.py       # fan-out wrapper over sandbox + alpaca_paper
│   ├── mcp_clients/
│   │   ├── __init__.py
│   │   ├── base.py              # subprocess + JSON-RPC lifecycle
│   │   ├── short_term.py
│   │   └── long_term.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── provider.py          # LLMProvider protocol + dataclasses
│   │   ├── github_models.py
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   ├── tool_loop.py
│   │   └── factory.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── policy.py            # shared risk gates + kill-switch
│   │   ├── day_trader.py
│   │   ├── long_term.py
│   │   └── coordinator.py
│   └── scheduler/
│       ├── __init__.py
│       └── runner.py            # APScheduler entry; `python -m src.scheduler.runner`
└── tests/
    ├── conftest.py
    ├── test_sandbox_engine.py
    ├── test_sandbox_broker.py
    ├── test_alpaca_broker.py    # via mocked MCP
    ├── test_mcp_clients.py      # contract tests
    ├── test_llm_tool_loop.py
    ├── test_policy.py
    ├── test_day_trader.py
    ├── test_long_term.py
    ├── test_coordinator.py
    └── test_scheduler.py
```

## 3. Data model (SQLite)

```sql
-- accounts: one row per sub-account. With DualBroker we keep parallel
-- accounts for each venue: 'day' + 'day_alpaca', 'long' + 'long_alpaca'.
-- 'master' is an optional ledger root for coordinator transfers.
CREATE TABLE accounts (
  id            INTEGER PRIMARY KEY,
  name          TEXT NOT NULL UNIQUE,         -- 'day'|'long'|'master'|'day_alpaca'|'long_alpaca'
  venue         TEXT NOT NULL,                -- 'sandbox'|'alpaca_paper'|'ledger'
  starting_cash REAL NOT NULL,
  created_at    TEXT NOT NULL                 -- ISO8601 UTC
);

-- cash_ledger: append-only cash movements; current cash = SUM(delta) per account
CREATE TABLE cash_ledger (
  id            INTEGER PRIMARY KEY,
  account_id    INTEGER NOT NULL REFERENCES accounts(id),
  ts            TEXT NOT NULL,
  delta         REAL NOT NULL,                -- +deposit / -withdraw / -buy / +sell
  reason        TEXT NOT NULL,                -- 'deposit'|'fill'|'fee'|'transfer'
  ref_order_id  INTEGER REFERENCES orders(id)
);

CREATE TABLE orders (
  id             INTEGER PRIMARY KEY,
  account_id     INTEGER NOT NULL REFERENCES accounts(id),
  ts             TEXT NOT NULL,                -- submission ts
  symbol         TEXT NOT NULL,
  side           TEXT NOT NULL,                -- 'buy'|'sell'
  qty            REAL NOT NULL,                -- fractional ok
  order_type     TEXT NOT NULL,                -- 'market'|'limit'
  limit_price    REAL,
  tif            TEXT NOT NULL,                -- 'day'|'gtc'
  status         TEXT NOT NULL,                -- 'pending'|'filled'|'cancelled'|'rejected'|'routed_external'
  submitted_at   TEXT NOT NULL,
  filled_at      TEXT,
  fill_price     REAL,
  fees           REAL NOT NULL DEFAULT 0,
  external_id    TEXT,                         -- alpaca order id when routed
  agent          TEXT NOT NULL,                -- 'day'|'long'|'coordinator'|'manual'
  thesis         TEXT,                         -- one-paragraph LLM rationale
  venue          TEXT NOT NULL,                -- 'sandbox'|'alpaca_paper'
  dual_group_id  TEXT                          -- shared UUID across the two legs of a DualBroker order
);
CREATE INDEX idx_orders_dual_group ON orders(dual_group_id);

CREATE TABLE positions_snapshot (
  id            INTEGER PRIMARY KEY,
  account_id    INTEGER NOT NULL REFERENCES accounts(id),
  ts            TEXT NOT NULL,
  symbol        TEXT NOT NULL,
  qty           REAL NOT NULL,
  avg_cost      REAL NOT NULL,
  mark_price    REAL NOT NULL
);

CREATE TABLE equity_curve (
  id              INTEGER PRIMARY KEY,
  account_id      INTEGER NOT NULL REFERENCES accounts(id),
  ts              TEXT NOT NULL,
  cash            REAL NOT NULL,
  positions_value REAL NOT NULL,
  equity          REAL NOT NULL                -- cash + positions_value
);

CREATE TABLE agent_runs (
  id            INTEGER PRIMARY KEY,
  account_id    INTEGER REFERENCES accounts(id),
  ts            TEXT NOT NULL,
  agent         TEXT NOT NULL,                -- 'day'|'long'|'coordinator'
  status        TEXT NOT NULL,                -- 'ok'|'halted'|'error'
  prompt        TEXT NOT NULL,
  response      TEXT,
  tools_called  TEXT,                          -- JSON array
  decisions     TEXT,                          -- JSON array of orders intended
  error         TEXT,
  latency_ms    INTEGER
);

-- UI -> scheduler RPC
CREATE TABLE tick_requests (
  id           INTEGER PRIMARY KEY,
  ts           TEXT NOT NULL,
  agent        TEXT NOT NULL,                  -- 'day'|'long'|'coordinator'
  requested_by TEXT NOT NULL DEFAULT 'ui',
  consumed_at  TEXT
);

CREATE TABLE settings (
  key          TEXT PRIMARY KEY,
  value        TEXT NOT NULL,
  updated_at   TEXT NOT NULL
);
-- known keys: 'kill_switch' ('on'|'off'),
--             'broker_backend' ('sandbox'|'alpaca_paper'),
--             'llm_provider' ('github'|'openai'|'anthropic'),
--             'llm_model',
--             'capital_total', 'split_day_pct'
```

Invariants:
- `cash(account) = SUM(cash_ledger.delta WHERE account_id = ?)` — always.
- Every `orders.status='filled'` row has exactly one matching
  `cash_ledger` row with `reason='fill'` and `ref_order_id = orders.id`.
- `equity_curve.equity = cash + positions_value` at each `ts`.
- `accounts.name` is one of `'master'`, `'day'`, `'long'`, `'day_alpaca'`,
  `'long_alpaca'`.
- Under `BROKER_BACKEND=dual`, every agent decision produces **exactly two**
  `orders` rows sharing the same `dual_group_id` — one with
  `venue='sandbox'`, one with `venue='alpaca_paper'`. Divergence in fill
  price/qty/status between the two legs is expected and is the whole point;
  it is recorded, never reconciled.

## 4. Broker interface

```python
# src/brokers/base.py
@dataclass
class AccountSnapshot:
    name: str
    equity: float
    cash: float
    positions_value: float

@dataclass
class Position:
    symbol: str
    qty: float
    avg_cost: float
    mark_price: float
    unrealized_pnl: float

@dataclass
class OrderRequest:
    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    order_type: Literal["market", "limit"] = "market"
    limit_price: float | None = None
    tif: Literal["day", "gtc"] = "day"
    agent: str = "manual"
    thesis: str | None = None

@dataclass
class OrderResult:
    id: int                # local orders.id
    external_id: str | None
    status: str
    fill_price: float | None
    fees: float

class BrokerBase(ABC):
    name: str
    def get_account(self) -> AccountSnapshot: ...
    def list_positions(self) -> list[Position]: ...
    def place_order(self, req: OrderRequest) -> OrderResult: ...
    def cancel_order(self, order_id: int) -> None: ...
    def close_position(self, symbol: str, percentage: float = 100.0) -> OrderResult: ...
    def mark_to_market(self, now: datetime) -> AccountSnapshot: ...
    def equity_curve(self, since: datetime | None = None) -> pd.DataFrame: ...
```

`SandboxBroker` enforces: blocklist, per-order USD cap, per-symbol % of
equity cap, no-shorting, cash sufficiency. `AlpacaPaperBroker` delegates to
the upstream `stock-recommender` MCP WRITE tools and mirrors every action
into our local `orders` table with `status='routed_external'` before reading
back the resolved status.

`DualBroker(primary=SandboxBroker, secondary=AlpacaPaperBroker)` is the
default when `BROKER_BACKEND=dual`. Contract:
- `place_order(req)` mints a `dual_group_id` (UUID4), then submits to both
  legs **sequentially** (primary first, then secondary). Both legs write
  their own `orders` row stamped with the same `dual_group_id` and the
  agent's `thesis`. Sequential (not parallel) because SQLite `Connection`
  is not safe across threads; per-trade latency cost is negligible. Sub-
  account routing: a `day`-agent order goes to `day` (sandbox) **and**
  `day_alpaca` (alpaca); same for `long`.
- If the **primary** raises, the call raises (the agent must see the failure).
  If only the **secondary** raises, the primary order stands and a
  `divergence` row is written (see below) — the secondary failure must not
  roll back the primary, because the sandbox is the source of truth for
  per-agent PnL.
- `cancel_order(order_id)` and `close_position(symbol, pct)` resolve the
  `dual_group_id` from the given row and cancel/close on both legs.
- Read methods (`get_account`, `list_positions`, `equity_curve`) return the
  **primary's** view by default; an explicit `venue=` kwarg returns the
  secondary's view. The Streamlit Overview tab calls both to plot two
  equity curves on shared axes.
- `mark_to_market(now)` is called on both legs every minute; both write into
  `equity_curve` against their own `account_id`.

Schema addition for divergence tracking:
```sql
CREATE TABLE dual_divergence (
  id            INTEGER PRIMARY KEY,
  dual_group_id TEXT NOT NULL,
  ts            TEXT NOT NULL,
  kind          TEXT NOT NULL,   -- 'secondary_error'|'fill_price'|'fill_qty'|'status'
  primary_val   TEXT,
  secondary_val TEXT,
  note          TEXT
);
CREATE INDEX idx_div_group ON dual_divergence(dual_group_id);
```

## 5. MCP client design

- One persistent `ClientSession` per upstream repo, owned by the scheduler
  process.
- `mcp_clients/base.py` provides:
  - `start()` → spawn `python -m mcp_server.server` in the sibling repo's
    `cwd` with merged env (we pass through Alpaca + Finnhub keys only when
    the user opted into `alpaca_paper`).
  - `call(tool, **args)` → JSON-RPC `tools/call`, 30 s timeout.
  - `health()` → ping with `tools/list`; auto-restart on consecutive failures.
- Contract tests in `tests/test_mcp_clients.py` assert the expected tool
  names + argument schemas; a failure means upstream drifted and the
  integration guide in `/memories/repo/` needs updating.
- If we need a tool that doesn't exist upstream, we **add a READ-only tool
  in the upstream repo** and capture the diff in `UPSTREAM_PATCHES.md`. We
  never touch upstream WRITE/safety code.

## 6. LLM layer

```python
# src/llm/provider.py
@dataclass
class ToolSpec:
    name: str
    description: str
    json_schema: dict

@dataclass
class ChatResult:
    text: str | None
    tool_calls: list[ToolCall]          # empty when text-only
    raw: dict                            # full provider response for the trace

class LLMProvider(Protocol):
    name: str
    model: str
    def chat(
        self,
        messages: list[dict],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ChatResult: ...
```

- `github_models.py` posts to `https://models.github.ai/inference/chat/completions`
  with `Authorization: Bearer ${GITHUB_TOKEN}`. Model defaults to
  `openai/gpt-4o-mini`. The endpoint speaks OpenAI-style function calling, so
  the OpenAI provider shares its tool-call adapter.
- `tool_loop.py` runs: model → if `tool_calls` → execute via the agent's tool
  registry → append `tool` role messages → loop. Bounded by `max_steps=8`.
  Every step (system prompt, user prompt, each tool call + result, final
  text) is captured for `agent_runs.tools_called`.
- `factory.py` resolves provider from the `settings` table first
  (`llm_provider`, `llm_model`), env second.

## 7. Agents

### 7.1 Day-Trader

Tools exposed to the LLM (subset, all read-only from the LLM's perspective —
the agent itself decides whether to place the order):

| LLM tool name | Backing call |
|---|---|
| `list_intraday_ideas` | `short_term.get_intraday_ideas(tier='A', min_heat=0.6)` |
| `get_quote` | `short_term.get_ticker_quote(symbol)` |
| `current_positions` | `broker.list_positions()` |
| `account_snapshot` | `broker.get_account()` |
| `propose_trade` | structured output → captured as a `Decision` |

After the LLM finishes, the agent applies hard gates (`policy.py`) and only
then calls `broker.place_order(...)` for each surviving `Decision`.

Hard rules (in code):
- Max 5 concurrent positions.
- Position size = `floor(0.01 * account.equity / (entry - stop))`.
- Hard stop submitted as a `sell limit` immediately after a fill.
- Force-flat at `15:55 ET` (cancel all open orders, market-close every long).
- Halt for the day if intraday drawdown ≤ `-2%` of starting equity.

### 7.2 Long-Term Investor

Tools:

| LLM tool name | Backing call |
|---|---|
| `get_recommendations` | `long_term.get_recommendations(universe='Curated', top_n=18)` |
| `get_portfolio_suggestion` | `long_term.get_portfolio_suggestion(budget=equity)` |
| `lookup_ticker` | `long_term.lookup_ticker(symbol)` |
| `get_news` | `long_term.get_news(symbol, days=14)` |
| `current_positions` | `broker.list_positions()` |
| `account_snapshot` | `broker.get_account()` |
| `propose_rebalance` | structured output |

Hard rules:
- One rebalance per 7 days unless drift > 10% on any symbol.
- Max 25% of sub-account equity per symbol.
- Inherits upstream blocklist (leveraged/inverse/vol ETFs).

### 7.3 Coordinator

No LLM call. Pure code:
- On the first trading day of the month, compute target cash for `day` and
  `long` sub-accounts from `capital_total * split_day_pct` and the inverse.
- Transfer cash via two `cash_ledger` rows (`reason='transfer'`) keeping the
  master account balanced.
- Refuse to transfer if either sub-account is currently in a kill-switched
  state.

### 7.4 Shared policy

- `kill_switch == 'on'` → every agent's `run_once` returns immediately with
  `agent_runs.status='halted'`.
- `policy.validate(decision, broker)` enforces caps and blocklist; rejected
  decisions still get written to `agent_runs.decisions` with a `reject_reason`.

## 8. Scheduler

`src/scheduler/runner.py` runs a `BlockingScheduler`:

| Job | Trigger | Action |
|---|---|---|
| `mtm` | every 1 min, mon–fri 09:30–16:00 ET | `broker.mark_to_market` for both sub-accounts |
| `day_tick` | every 5 min, mon–fri 09:30–15:55 ET | `DayTraderAgent.run_once` |
| `long_tick` | daily 16:30 ET on trading days | `LongTermAgent.run_once` |
| `coord_tick` | first trading day of month 16:45 ET | `Coordinator.run_once` |
| `tick_poll` | every 5 s, always | drain `tick_requests`, run requested agent immediately |

A `data/scheduler.lock` file prevents double-start.

## 9. Streamlit UI

- `app.py` opens SQLite read-only and renders the six tabs in `plan.md`
  Phase 7.
- "Tick now" buttons insert into `tick_requests`; the scheduler picks them up
  within ~5 s.
- Settings changes write to the `settings` table; the scheduler re-reads
  `settings` at the top of every job tick.

## 10. Configuration

`src/config.py` uses `pydantic-settings`. Resolved order: function arg →
`settings` table → `.streamlit/secrets.toml` → env var → default.

Required:
- `GITHUB_TOKEN` (if `LLM_PROVIDER=github`)
- `OPENAI_API_KEY` (if `LLM_PROVIDER=openai`)
- `ANTHROPIC_API_KEY` (if `LLM_PROVIDER=anthropic`)
- `SHORT_TERM_TRADER_PATH` — absolute path to sibling repo
- `STOCK_RECOMMENDER_PATH` — absolute path to sibling repo

Required when `BROKER_BACKEND` is `alpaca_paper` **or** `dual`:
- `ALPACA_API_KEY_ID`, `ALPACA_SECRET_KEY`, `ALPACA_PAPER=true`,
  `STOCK_REC_MCP_TRADING_ENABLED=true`.

## 11. Safety guarantees

- Sandbox broker can never reach the internet; all "fills" come from local
  yfinance bars cached on disk.
- Alpaca paper broker hard-asserts `ALPACA_PAPER=true` and that the account
  number starts with `PA` (delegated to upstream).
- Global blocklist; per-order USD cap; per-symbol % cap.
- Kill-switch in `settings` table halts every agent in ≤ one tick.
- Every order is double-logged: in `orders` and (if routed externally) in the
  upstream's own audit log.
- In `dual` mode, the **sandbox leg is the source of truth** for agent
  decisions and per-agent PnL; the Alpaca leg is a verification mirror. A
  failure on the Alpaca side never corrupts the sandbox ledger.

## 12. Testing strategy

- **Unit:** fill engine determinism, cash conservation, MTM math, policy gates.
- **Integration:** one full tick of each agent against a fake `LLMProvider`
  that emits scripted tool calls + a fake `MCPClient` returning canned ideas.
- **Contract:** spawn each real upstream MCP server (skipped if path env
  unset), call `tools/list`, assert the names + schemas we depend on.
- **Smoke (manual):** `bash run.sh`, click through each tab.

## 13. Deployment

- Local: `bash run.sh` starts scheduler in background, Streamlit in foreground.
- Docker: `docker compose up` builds one image, runs two services
  (`scheduler`, `web`) sharing the `data/` volume; both sibling repos are
  bind-mounted read-only so MCP subprocesses can launch.
- No cloud deploy in v1.
