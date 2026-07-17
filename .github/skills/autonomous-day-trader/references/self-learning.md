# Self-Learning Loop

The point of this skill is to trade **and get better at it over time**, the way a
human trader keeps a journal and reviews it. Three artifacts make that work:

| File | What it is | Written by | Read when |
|---|---|---|---|
| `data/reasoning_log.jsonl` | Every agent decision + cited data (auto) | the agent, on every run | analysis / review |
| `data/trader_journal.jsonl` | Your observations, actions, lessons | `journal_append.py`, `end_of_day_review.py` | any check-in |
| `data/strategy_notes.md` | Distilled, durable lessons that shape decisions | you (curated) | **every session start** |

The loop: **observe → journal → act → review → distil into strategy_notes →
apply next session.**

## `data/reasoning_log.jsonl` (automatic)

One JSON line per decision, already produced by the agent. Schema:

```json
{"run_id": 5461, "ts": "2026-07-16T16:50:16Z", "agent": "day",
 "rationale": "Exited the weak option; proposed buy AAPL ...",
 "data_sources": ["list_intraday_ideas", "get_news:AAPL", "get_analyst_view:AAPL"],
 "symbol": "AAPL", "action": "buy", "qty": 10, "thesis": "...", "accepted": true}
```

Re-export/backfill anytime: `python -m scripts.export_reasoning`. This is the raw
material — pairs of (reasoning + data) → (trade) you can replay to see what led to
good vs bad outcomes.

## `data/trader_journal.jsonl` (you, continuously)

Append-only, one JSON line per entry, via `journal_append.py`. Schema:

```json
{"ts": "2026-07-16T14:32:00Z", "kind": "observation",
 "summary": "Semis leading tape down; idea pool 60% semis shorts",
 "detail": "AMAT/MU/MRVL tier-A shorts. Watch for reversal into power hour.",
 "tags": ["semis", "regime"], "refs": {"run_id": 5461, "symbols": ["AMAT"]}}
```

`kind` values and when to write them:

- `session_open` — start of day: bias, watch-list themes, plan, and which
  lessons from `strategy_notes.md` you're carrying in.
- `observation` — a market/agent read the mechanical loop can't judge: a regime
  shift, a crowded one-way book, a catalyst it's ignoring, repeated throttling.
- `action` — a toggle you flipped (compact mode, kill-switch, universe) and why.
- `trade` — a discretionary entry/exit you made, with the thesis and the stop.
- `lesson` — an in-the-moment insight worth remembering.
- `session_close` — the end-of-day review (written by `end_of_day_review.py`).

Write **honestly and specifically**. "Shorted AMAT, wrong, tape reversed at
2 pm — stop was too tight for the ATR" teaches more than "bad trade."

## `data/strategy_notes.md` (you, curated)

The distilled playbook — the *learning* that persists. After each end-of-day
review, promote 1–3 durable lessons here. Keep it **short and bulleted**; it is
loaded at every session start and is what makes the agent improve. Example:

```markdown
# Strategy notes (learned)

## Regime / setups
- Semis-heavy idea pools on down days → shorts crowd; fade only with a catalyst,
  size smaller, and prefer the inverse ETF for cleaner fills.
- Tier-A with no news catalyst underperforms — demand stronger structure.

## Risk
- 5% default stop is too tight for high-ATR names (SOXL/TSLZ) — widen to 1.5×ATR.
- Book fills to 8 by midday → rotate weakest by 11:00 ET, don't wait for stops.

## Ops
- Enable compact mode by 9:35 ET on days gpt-5 is throttled; it keeps ticks alive.
- sp500 universe surfaces more non-semis names than liquid — keep it.
```

## The daily loop (concrete)

1. **Session start:** read `strategy_notes.md` + tail of the journal → journal a
   `session_open` with today's plan.
2. **Intraday check-ins:** run `session_snapshot.py`; journal `observation`s and
   `action`s; take/adjust trades per [trading-rules.md](./trading-rules.md).
3. **After close (flat):** run `end_of_day_review.py` → it appends a
   `session_close` review summarizing fills, P&L, decisions, throttle time.
4. **Distil:** read the review + the day's `reasoning_log.jsonl`; write 1–3
   concrete lessons into `strategy_notes.md`. Prune stale/º wrong notes.

## What to actually learn (prompts for reflection)

- Which setups/tiers/catalysts led to profitable vs losing trades today?
- Were stops hit for real invalidation, or noise (too tight for the ATR)?
- Did the book over-concentrate in one theme? Did rotation help or churn fees?
- Did throttling cost trades? Would compact mode earlier have helped?
- Did discretionary actions beat or lag the mechanical agent? Be honest.
- Is the idea universe surfacing enough variety, or the same narrow set?

Turn the answers into edits to `strategy_notes.md`. Over weeks this file becomes
a genuine, evidence-based trading playbook — the system "learning like a human."
