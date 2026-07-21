---
name: autonomous-day-trader
description: 'Run an autonomous, self-learning PAPER day-trader where COPILOT ITSELF is the trading brain (no GitHub Models PAT / no external LLM API call). USE FOR: gathering ranked intraday ideas + news + analyst views as one context dump; reasoning over them in Copilot; placing/exiting PAPER trades on Alpaca via an LLM-free execution script (sizing, caps, stops, logging all applied); journaling observations/actions/trades/lessons to a durable log; end-of-day review that evolves a strategy playbook; toggling autopilot (Copilot-driven vs scheduler-LLM), compact mode, kill-switch. Goal: maximize daily risk-adjusted return while preserving capital. PAPER/SIMULATED ONLY \u2014 no live-money path. DO NOT USE FOR: live/real-money trading; unrelated coding tasks.'
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
service** — it monitors stops, refreshes ideas, and reconciles orders on its
own, all day. **In COPILOT-DRIVEN mode (the intended mode for this skill),
Copilot is the trading brain** — the scheduler makes NO LLM/API calls, and you
(the model running in Copilot) do the reasoning each minute, so the GitHub
Models PAT is never hit and there's no rate-limit ceiling on decisions.

**Your role via this skill is twofold:**

1. **Trade** — each minute, read the market context, decide, and act (place /
   exit trades) using the LLM-free execution scripts. Copilot's own model does
   the reasoning; the engine applies sizing, caps, stops, and logging.
2. **Learn** — journal observations, actions, trades and lessons, and evolve
   `data/strategy_notes.md` so decisions keep improving over time.
> VS Code Copilot is turn-based and cannot literally self-trigger every 60 s
> while fully unattended. **The skill's autonomous loop is COPILOT ITSELF
> iterating** the gather→reason→act cycle within a working session (see "The
> loop" below) — that is what's different from the existing scheduler agent, and
> it uses no PAT. Between your turns the scheduler's LLM-free jobs (stop monitor
> every ~30 s, end-of-day force-flat, scan refresh, reconcile) protect the book.
> Keep the `scheduler` container running.

## Autonomy modes (`day_autopilot`, live toggle — no rebuild)

The existing scheduler agent and this skill are **separate**, chosen by one
setting so they never both trade at once:

- **`off` — Copilot-driven (this skill):** the scheduler makes NO LLM/PAT calls;
  **Copilot is the brain** — you loop over `gather_context.py` → reason →
  `execute_trade.py` each minute. This is the skill's autonomous mode.
- **`on` — the existing scheduler agent:** the scheduler's `day_tick` calls the
  configured LLM (GitHub Models / OpenAI / Anthropic) and trades on its own —
  the classic engine, unchanged. Uses the PAT; bounded by that provider's rate
  limits.

Set the mode:
```bash
docker compose exec -T scheduler python -c \
 "from src.sandbox import db as d; from src.config import db_path; \
  d.set_setting(d.get_conn(db_path()),'day_autopilot','off')"   # 'off' = skill/Copilot ; 'on' = scheduler agent
```

Use `off` while you're running the skill (Copilot trades); use `on` to hand
control back to the standalone scheduler agent when no Copilot session is open.

## Procedure

### 0. Session start (once per trading day)

1. Verify secrets exist and keys are wired (see
   [architecture-and-setup.md](./references/architecture-and-setup.md) for the
   full env-var table and the news/Alpaca API keys the skill is allowed to use).
   In particular, `COPILOT_SKILLS_ALPACA_API_KEY_ID` /
   `COPILOT_SKILLS_ALPACA_SECRET_KEY` must be set in
   `.streamlit/secrets.toml` — these bind the skill to its **dedicated paper
   account `PA3WWRZG806F`** (separate from the scheduler agent's account). Every
   skill script hard-aborts unless the connected `account_number` matches, so it
   can never trade the wrong book.
2. Start the stack: `./trader start` (or `./trader rebuild` after code changes —
   **`restart` does NOT reload code**; the image bakes `app.py`/`src/`).
3. **Enter Copilot-driven mode** so the standalone scheduler agent stands down
   and the skill (Copilot) is the trader — no PAT calls:
   `docker compose exec -T scheduler python -c "from src.sandbox import db as d; from src.config import db_path; d.set_setting(d.get_conn(db_path()),'day_autopilot','off')"`
4. Confirm health + market state with the snapshot:
   `docker compose exec -T scheduler python .github/skills/autonomous-day-trader/scripts/session_snapshot.py`
5. Read yesterday's lessons: open `data/strategy_notes.md` and the tail of
   `data/trader_journal.jsonl`. Carry forward what worked / what to avoid.
6. Journal a session-open entry (bias, watch-list themes, plan) with
   `scripts/journal_append.py` (see step "Journal").

### 1. The loop — keep trading without intervention (Copilot iterates this)

**This is the skill's autonomous loop: YOU (Copilot) repeat steps a–d below,
once per minute, for as long as your working session runs.** Each pass is a full
trade decision made by Copilot's own model — no PAT, no external LLM call. After
acting, wait ~60 s (or re-run when prompted) and go again. The scheduler's stop
monitor protects open positions between passes and force-flattens at 15:55 ET.

**a. Gather** — one JSON dump of everything you need to decide (market state,
account, current book with per-holding P&L + distance-to-stop, and the top
ranked ideas each enriched with news sentiment + analyst view — the REQUIRED
diligence):

```bash
docker compose exec -T scheduler python \
  .github/skills/autonomous-day-trader/scripts/gather_context.py --ideas 6 --news 6
```

**b. Reason (you, Copilot — this is the "LLM", no PAT).** Apply the discipline in
[trading-rules.md](./references/trading-rules.md): pick 0–3 best setups with a
news + analyst cross-check, a defined stop, and ≥ 2:1 R:R; if `book_full`, choose
the weakest holding to exit first. Also review holdings — cut losers / invalidated
theses.

**c. Act** — for each decision, call the LLM-free executor (it applies inverse-ETF
substitution, price sanity, sizing, caps, the live stop plan and audit logging):

```bash
# open a position
docker compose exec -T scheduler python \
  .github/skills/autonomous-day-trader/scripts/execute_trade.py --enter \
  --symbol AAPL --side buy --entry 150.00 --stop 143.00 --target 164.00 \
  --thesis "Breakout > VWAP; positive news + Buy rating; 2:1 R:R"

# free a slot / cut a loser / take profit
docker compose exec -T scheduler python \
  .github/skills/autonomous-day-trader/scripts/execute_trade.py --exit \
  --symbol PINS --reason "weakest holding; thesis stalled; rotating"
```

**d. Journal** the observation/action (see "Journal"). Then repeat next minute.

**e. Report (every phase).** Render the per-phase status tables so the book and
what just happened are always visible:

```bash
docker compose exec -T scheduler python \
  .github/skills/autonomous-day-trader/scripts/status_table.py --activity 8
```

This prints a **Holdings** table (symbol, side, qty, avg, mark, P&L, stop,
%-to-stop, %-equity, theme) and a **Recent activity** table (last N journal
entries). Always show both tables at the start (post-gather) and end (post-act)
of each pass.

If nothing qualifies, do nothing — journal a brief `observation` and move on.

### 2. Trade discipline (always)

Follow [trading-rules.md](./references/trading-rules.md) — non-negotiable and
mirrors the engine's enforced rules:
- **Diligence gate:** never enter without checking news **and** analyst view for
  that symbol (both are in the `gather_context.py` output per idea).
- **Defined risk:** every entry needs an explicit stop and ≥ 2:1 reward:risk.
- **Caps:** ≤ ~20% equity per name, ≤ ~35% per correlation theme, ≤ 8 concurrent
  positions. If the book is full, **exit the weakest** before adding.
- **Time-of-day:** no new entries after 15:30 ET; everything auto-flattens 15:55 ET.

`execute_trade.py` runs the same sizing/caps/stop/logging engine as an autonomous
tick — just with Copilot as the brain instead of an API call. It still rejects a
trade that breaks a cap (returns `rejected` with a reason). See
[trading-rules.md](./references/trading-rules.md) for the full command reference.

### 2b. Smart-money edge (insider C-suite + political) — optional

When an `FMP_API_KEY` (recommended — covers both) or `FINNHUB_API_KEY` (insider
only) is configured, each idea in `gather_context.py` carries a `smart_money`
block (overall/insider/political bias + net disclosed $ flow) and the dump
includes a market-wide `smart_money_activity` ranking. Use it as a **conviction
tilt only** — a fresh cluster of insider/political **buys** supports a long,
heavy insider **selling** is a caution flag — never as a standalone trigger; the
news + analyst + technical gate still applies. On demand:

```bash
# stocks with unusual insider/political activity right now
docker compose exec -T scheduler python \
  .github/skills/autonomous-day-trader/scripts/smart_money_report.py --market

# deep-dive one symbol
docker compose exec -T scheduler python \
  .github/skills/autonomous-day-trader/scripts/smart_money_report.py --symbol NVDA

# approximate a person's net positions across stocks + sectors
docker compose exec -T scheduler python \
  .github/skills/autonomous-day-trader/scripts/smart_money_report.py --person "Pelosi"
```

Positions are reconstructed from disclosed **transactions** over the lookback
(insiders/politicians don't publish real-time holdings), so treat them as
"recent net activity", not a brokerage statement. No key ⇒ the block is simply
absent and nothing breaks.


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

- **Autopilot** (`day_autopilot`): `off` = Copilot-driven (this skill trades; the
  standalone scheduler agent stands down; no PAT). `on` = the existing scheduler
  agent trades on its own via the LLM (uses the PAT). Use `off` while running the
  skill, `on` to hand control back to the standalone agent.
- **Compact mode** (`compact_prompt`): only relevant in autopilot `on` mode —
  turn ON when the scheduler's LLM is throttled (429/413) so requests fit the
  free-tier fallback model. Irrelevant in Copilot-driven mode (no PAT calls).
- **Kill switch** (`kill_switch`): set `on` to halt all agents within one tick
  (positions untouched). Use if something looks wrong.
- **Scan universe** (`SCAN_UNIVERSE` env: `sp500` default, `liquid`, `all`) and
  cadence — see [architecture-and-setup.md](./references/architecture-and-setup.md).

## Guardrails you must respect

- **Paper only.** Never add a live-broker path. Alpaca stays paper
  (`ALPACA_PAPER=true`, account starts with `PA`).
- **Dedicated account.** The skill trades only account **`PA3WWRZG806F`** via
  `COPILOT_SKILLS_ALPACA_*` (an override applied inside each skill process, so
  the scheduler agent's account is untouched). `_trader.py` asserts the
  connected `account_number` equals `PA3WWRZG806F` and aborts otherwise.
- **Dedicated sub-account namespace.** The skill uses
  `COPILOT_SKILLS_SUB_ACCOUNT=copilot_skills` (inside `_trader.py`) so its
  DB stop plans / runs are isolated from the standalone scheduler agent's
  `day` namespace. This prevents cross-pruning of active stop plans.
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

- [gather_context.py](./scripts/gather_context.py) — **Copilot's eyes**: one JSON
  dump of market state + account + book (P&L, distance-to-stop) + top ranked
  ideas enriched with news + analyst (read-only; no LLM/PAT call).
- [execute_trade.py](./scripts/execute_trade.py) — **Copilot's hands**: place
  (`--enter`) or close (`--exit`) a trade through the real engine (sizing, caps,
  inverse-ETF substitution, live stop plan, audit logging) — no LLM call.
- [session_snapshot.py](./scripts/session_snapshot.py) — health + positions +
  today's decisions + P&L + throttle state (read-only).
- [journal_append.py](./scripts/journal_append.py) — append a structured entry
  to `data/trader_journal.jsonl`.
- [status_table.py](./scripts/status_table.py) — **per-phase report**: markdown
  **Holdings** table (P&L, stop, distance-to-stop, %-equity, theme) + **Recent
  activity** table (last N journal entries). Show it every pass (read-only).
- [end_of_day_review.py](./scripts/end_of_day_review.py) — aggregate the day and
  write a `session_close` review.
