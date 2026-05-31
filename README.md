# agent-trader-sandbox

An LLM-driven, paper-only agentic trader with a built-in sandbox.

- **Day-Trader agent** consumes intraday signals from the sibling
  [`short-term-trader`](../short-term-trader) project (via MCP).
- **Long-Term Investor agent** consumes recommendations from the sibling
  [`stock-recommender`](../stock-recommender) project (via MCP).
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

## Repository layout

See [`design.md`](design.md) §2.

## Documentation

- [`plan.md`](plan.md) — phases, verification, decisions
- [`design.md`](design.md) — architecture, data model, broker interface
- [`tasks.md`](tasks.md) — per-phase checklist
- [`AGENTS.md`](AGENTS.md) — sandbox safety guarantees and env vars
- [`UPSTREAM_PATCHES.md`](UPSTREAM_PATCHES.md) — diffs applied to sibling repos
