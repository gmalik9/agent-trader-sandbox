# agent-trader-sandbox

An LLM-driven, paper-only agentic trader with a built-in sandbox.

Live app: https://agent-trader.streamlit.app/
Repo: https://github.com/gmalik9/agent-trader-sandbox

### Companion projects

| Project | Repo | Live app |
|---|---|---|
| Short-Term Trader (intraday signals MCP) | https://github.com/gmalik9/short-term-stock-recommender | https://short-term-stock.streamlit.app/ |
| Long-Term Stock Recommender (Alpaca paper MCP) | https://github.com/gmalik9/long-term-stock-recommender | https://long-term-stock.streamlit.app/ |

- **Day-Trader agent** consumes intraday signals from the sibling
  [short-term-stock-recommender](https://github.com/gmalik9/short-term-stock-recommender) project (via MCP).
- **Long-Term Investor agent** consumes recommendations from the sibling
  [long-term-stock-recommender](https://github.com/gmalik9/long-term-stock-recommender) project (via MCP).
- **Coordinator** splits a configurable bankroll between the two sub-accounts.
- **DualBroker** (default) executes every order on the local sandbox **and**
  on Alpaca paper in parallel, so you can compare the agent's local PnL
  against an independent venue.
- **Streamlit dashboard** for equity curves, blotter, positions, PnL, and the
  agents' reasoning trace.

> Paper / simulated only. No live-money path exists in this repo.

## Quickstart

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit secrets.toml: set GITHUB_TOKEN, Alpaca keys, sibling-repo paths

bash run.sh
# open http://localhost:8501
```

Or with Docker:

```bash
docker compose up --build
```

## Local always-on trading

For unattended 24/7 ticking (independent of whether the Streamlit tab is open),
run the dedicated scheduler process. It uses the **real** sibling MCP servers
when `SHORT_TERM_TRADER_PATH` / `STOCK_RECOMMENDER_PATH` are configured, and
falls back per-leg to the built-in yfinance provider if a server can't start.

```bash
# Headless, auto-restarting scheduler (no UI). Ctrl-C to stop.
./scheduler.sh
# detach + log:  nohup ./scheduler.sh & ; tail -f logs/scheduler.log
```

Or via Docker (the `scheduler` service has `restart: unless-stopped`):

```bash
docker compose up -d scheduler      # scheduler only
docker compose up -d                # scheduler + dashboard on :8501
```

> **One scheduler at a time.** `run.sh`, `scheduler.sh`, and the Docker
> `scheduler` service each own all ticking. The Streamlit app can *also* run an
> in-process scheduler (used on Streamlit Cloud). To avoid double-ticking the
> same book, set `INPROCESS_SCHEDULER=0` for the dashboard whenever a dedicated
> scheduler is already running — `run.sh` and the Docker `web` service do this
> automatically.

### About the deployed sibling apps

The companion projects are deployed as Streamlit **dashboards**
([long-term](https://long-term-stock.streamlit.app/) /
[short-term](https://short-term-stock.streamlit.app/)). Streamlit Cloud only
runs each repo's `app.py`, **not** their MCP servers or FastAPI sidecars, so
those public URLs can't be used as data endpoints for this agent. Signal
sourcing is therefore: real sibling MCP subprocesses when running locally with
the repos checked out, and the built-in yfinance fallback otherwise (including
on Streamlit Cloud).

## Streamlit Cloud

The app also runs on Streamlit Cloud as a single process: APScheduler is
started in a background thread on first render so the agents tick without a
separate scheduler service. Deploy by pointing Cloud at `app.py`, then add
`GITHUB_TOKEN` (and any optional Alpaca/Finnhub keys) in the Cloud "Secrets"
panel. The sibling-repo MCP servers are not available on Cloud, so the
broker automatically falls back to `BROKER_BACKEND=sandbox` (yfinance fills,
no Alpaca mirror) — set the variables in `.streamlit/secrets.toml.example`
as a starting point.

### Trade ideas without the sibling MCP servers

The Day-Trader and Long-Term agents normally source ideas, quotes, and
recommendations from the two sibling repos launched as local MCP subprocesses.
Those repos don't exist on Streamlit Cloud, so when `SHORT_TERM_TRADER_PATH` /
`STOCK_RECOMMENDER_PATH` are unset (or the subprocess can't start), the
scheduler transparently swaps in a self-contained **yfinance-only fallback**
(`src/signals/local.py`). It produces real momentum/trend ideas and prices, so
the agents keep trading the sandbox book. Because the fallback can't reach the
Alpaca write tools, the Alpaca paper mirror stays disabled in that mode.

> Streamlit Cloud sleeps idle apps, which pauses the in-process scheduler.
> While the app is open the agents tick on schedule (day ticks are gated to
> market hours); you can also force a run anytime from the **Settings** tab's
> "Tick now" buttons — the Long-Term agent trades immediately.

## Repository layout

See [`design.md`](design.md) §2.

## Documentation

- [`plan.md`](plan.md) — phases, verification, decisions
- [`design.md`](design.md) — architecture, data model, broker interface
- [`tasks.md`](tasks.md) — per-phase checklist
- [`AGENTS.md`](AGENTS.md) — sandbox safety guarantees and env vars
- [`UPSTREAM_PATCHES.md`](UPSTREAM_PATCHES.md) — diffs applied to sibling repos
