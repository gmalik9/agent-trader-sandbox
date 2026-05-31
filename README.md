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

## Streamlit Cloud

The app also runs on Streamlit Cloud as a single process: APScheduler is
started in a background thread on first render so the agents tick without a
separate scheduler service. Deploy by pointing Cloud at `app.py`, then add
`GITHUB_TOKEN` (and any optional Alpaca/Finnhub keys) in the Cloud "Secrets"
panel. The sibling-repo MCP servers are not available on Cloud, so the
broker automatically falls back to `BROKER_BACKEND=sandbox` (yfinance fills,
no Alpaca mirror) — set the variables in `.streamlit/secrets.toml.example`
as a starting point.

## Repository layout

See [`design.md`](design.md) §2.

## Documentation

- [`plan.md`](plan.md) — phases, verification, decisions
- [`design.md`](design.md) — architecture, data model, broker interface
- [`tasks.md`](tasks.md) — per-phase checklist
- [`AGENTS.md`](AGENTS.md) — sandbox safety guarantees and env vars
- [`UPSTREAM_PATCHES.md`](UPSTREAM_PATCHES.md) — diffs applied to sibling repos
