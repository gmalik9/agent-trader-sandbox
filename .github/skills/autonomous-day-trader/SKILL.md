---
name: autonomous-day-trader
description: 'Run and supervise an autonomous, self-learning PAPER day-trading agent on Alpaca. USE FOR: starting/monitoring the day-trading engine; pulling ranked intraday ideas + news + analyst views; placing/exiting PAPER trades on Alpaca; journaling market observations, actions and trades to a durable learning log; running an end-of-day review that extracts lessons and evolves a strategy playbook; tuning runtime toggles (compact mode, scan universe, kill-switch). Goal: maximize daily risk-adjusted return while preserving capital. PAPER/SIMULATED ONLY — there is no live-money path. DO NOT USE FOR: live/real-money trading; unrelated coding tasks.'
argument-hint: '[start | check | trade | journal | review]'
---

# Autonomous Day Trader (paper, self-learning)

Operate a disciplined intraday trading agent on an **Alpaca paper account**, and
learn from every session like a human trader would — journaling observations,
the actions taken, the trades made and their outcomes, then distilling lessons
that shape future decisions. **Objective: maximize daily risk-adjusted return
while protecting capital.**

> ⚠️ **Paper / simulated only.** This skill trades an Alpaca *paper* account.
> There is no live-money code path anywhere in this repo, and you must never
> create one. All guarantees are in [AGENTS.md](../../../AGENTS.md).

## What this system is (read once)

Three cooperating repos (see [architecture-and-setup.md](./references/architecture-and-setup.md)):

- **agentic-trader** (this repo) — the trader. A `DayTraderAgent` runs every
  ~2 min during market hours; a stop-monitor runs every ~30 s. Orders go to
  **Alpaca paper (primary)** and mirror to a local sandbox. Every decision is
  logged to SQLite (`data/sandbox.sqlite`) and appended to
  `data/reasoning_log.jsonl`.
- **short-term-trader** (sibling, read-only) — the intraday **scanner**: ranked
  trade ideas (`list_ideas`), quotes (`lookup_ticker`), and 9-source news
  sentiment (`get_news`). Consumed via an MCP subprocess.
- **stock-recommender** (sibling) — the **Alpaca paper broker MCP** (account,
  positions, order placement, close) + analyst views. Consumed via MCP.

The **truly autonomous background execution is the Docker `scheduler`
service** — it ticks the agent and monitors stops on its own, all day, no human
needed. **Your role via this skill is twofold:**

1. **Supervise** the engine — start it, confirm it's healthy and trading, tune
   runtime toggles when the market or LLM quota demands it.
2. **Learn** — periodically review what happened, journal observations and
   lessons, and evolve `data/strategy_notes.md` so the agent keeps improving.

> VS Code Copilot is turn-based and cannot itself sit unattended for 6.5 hours.
> The scheduler provides the unattended execution; this skill makes each of your
> check-ins productive and cumulative. For true 24/7 autonomy, keep the
> `scheduler` container running.

## Autonomy modes

- **Supervised-autonomous (recommended):** the `scheduler` container trades
  continuously; you check in periodically (e.g. open, midday, power hour, close)
  to review, journal, and tune. This is what "runs all day without
  intervention" means in practice.
- **Direct:** within a session, drive individual ticks and place/exit trades
  yourself, journaling as you go. Use for debugging or hands-on trading.

## Procedure

### 0. Session start (once per trading day)

1. Verify secrets exist and keys are wired (see
   [architecture-and-setup.md](./references/architecture-and-setup.md) for the
   full env-var table and the news/Alpaca API keys the skill is allowed to use).
2. Start the stack: `./trader start` (or `./trader rebuild` after code changes —
   **`restart` does NOT reload code**; the image bakes `app.py`/`src/`).
3. Confirm health + market state with the snapshot:
   `docker compose exec -T scheduler python .github/skills/autonomous-day-trader/scripts/session_snapshot.py`
4. Read yesterday's lessons: open `data/strategy_notes.md` and the tail of
   `data/trader_journal.jsonl`. Carry forward what worked / what to avoid.
5. Journal a session-open entry (bias, watch-list themes, plan) with
   `scripts/journal_append.py` (see step "Journal").

### 1. Check-in (every ~30–60 min while the market is open)

Run the snapshot script and read it critically:
- Is the agent ticking (recent `ok` runs)? Is it throttled (`rate_limited` /
  `413`)? If throttled, enable **compact mode** (see tuning below).
- Current positions, per-name %, gross exposure, unrealized P&L.
- Today's decisions and any rejects (`insufficient_diligence`, `theme_at_cap`,
  `max_concurrent_positions`, `size_rounded_to_zero`).
- Stops: are open positions protected (active `position_plans`)?

If you spot something the mechanical agent can't judge (a regime shift, a
crowded one-way book, a news catalyst it's ignoring), **journal the observation**
and, if warranted, act (place/exit a trade, or tune a toggle).

### 2. Trade (when you take a discretionary action)

Follow the discipline in [trading-rules.md](./references/trading-rules.md) —
this is non-negotiable and mirrors the enforced rules:
- **Diligence gate:** never enter without checking news **and** analyst view for
  that symbol.
- **Defined risk:** every entry needs an explicit stop and ≥ 2:1 reward:risk.
- **Caps:** ≤ ~20% equity per name, ≤ ~35% per correlation theme, ≤ 8 concurrent
  positions. If the book is full, **exit the weakest** before adding.
- **Time-of-day:** no new entries after 15:30 ET; everything auto-flattens 15:55 ET.

Place/exit through the running agent's broker (it applies stops, caps, and
logging automatically). Prefer queuing a tick or using the existing tools over
raw Alpaca calls; see [trading-rules.md](./references/trading-rules.md) for the
exact commands.

### 3. Journal (continuously — this is the "self-learning")

After every meaningful observation or action, append a structured entry:

```bash
docker compose exec -T scheduler python \
  .github/skills/autonomous-day-trader/scripts/journal_append.py \
  --kind observation \
  --summary "Semis leading tape down; scanner idea pool 60% semis shorts" \
  --detail "AMAT/MU/MRVL all tier-A shorts. Watching for a reversal into power hour." \
  --tags semis,regime
```

`--kind` ∈ `observation | action | trade | lesson | session_open | session_close`.
Entries land in `data/trader_journal.jsonl` (append-only, one JSON line each).
See [self-learning.md](./references/self-learning.md) for the full schema and the
learning loop.

### 4. End-of-day review (once, after 15:55 ET / flat)

```bash
docker compose exec -T scheduler python \
  .github/skills/autonomous-day-trader/scripts/end_of_day_review.py
```

This aggregates the day (fills, realized P&L, decision counts, throttle time,
rejects) and writes a `session_close` review to the journal. Then **you** read
it and distil 1–3 durable lessons into `data/strategy_notes.md` — concise,
bulleted, and honest (what worked, what didn't, what to change tomorrow). This
file is loaded at every session start, so it is how the agent "learns like a
human over time."

## Runtime tuning (no rebuild — live toggles in SQLite `settings`)

- **Compact mode** (`compact_prompt`): turn ON when the LLM is throttled (429/413)
  so requests fit the free-tier fallback model and the agent keeps trading. Flip
  in the Streamlit **Settings** tab, or:
  `docker compose exec -T scheduler python -c "from src.sandbox import db as d; from src.config import db_path; d.set_setting(d.get_conn(db_path()),'compact_prompt','on')"`
- **Kill switch** (`kill_switch`): set `on` to halt all agents within one tick
  (positions untouched). Use if something looks wrong.
- **Scan universe** (`SCAN_UNIVERSE` env: `sp500` default, `liquid`, `all`) and
  cadence — see [architecture-and-setup.md](./references/architecture-and-setup.md).

## Guardrails you must respect

- **Paper only.** Never add a live-broker path. Alpaca stays paper
  (`ALPACA_PAPER=true`, account starts with `PA`).
- **Read-only DB inspection.** `data/sandbox.sqlite` is WAL-mode and the
  scheduler writes to it constantly. Inspect with a **read-only** connection
  (`file:...?mode=ro`) or the snapshot script. Never run concurrent writers — it
  corrupted the DB once.
- **Deploy = rebuild.** `docker compose up -d --build <svc>` (or `./trader
  rebuild`) after editing `.py`/`app.py`. `restart` alone serves stale code.
- **Secrets stay in `.streamlit/secrets.toml`** (git-ignored). The skill may
  read the news-source and Alpaca keys from there and from the container env to
  do its job; never print secrets or commit them.
- **Options can't be closed via the stock endpoint** — they expire on their own.

## Reference files

- [architecture-and-setup.md](./references/architecture-and-setup.md) — the
  3-repo architecture, env vars, API keys, how to run, health checks.
- [trading-rules.md](./references/trading-rules.md) — the trading discipline and
  the exact commands to pull ideas / place / exit / monitor.
- [self-learning.md](./references/self-learning.md) — journal schema, the
  learning loop, and how `strategy_notes.md` feeds back into decisions.

## Scripts

- [session_snapshot.py](./scripts/session_snapshot.py) — health + positions +
  today's decisions + P&L + throttle state (read-only).
- [journal_append.py](./scripts/journal_append.py) — append a structured entry
  to `data/trader_journal.jsonl`.
- [end_of_day_review.py](./scripts/end_of_day_review.py) — aggregate the day and
  write a `session_close` review.
