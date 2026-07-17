# Trading Rules (the discipline)

These mirror the rules the `DayTraderAgent` enforces mechanically. When you take
a discretionary action, follow them too — they are how the system protects
capital while chasing return.

## Objective

Maximize **daily risk-adjusted return** while preserving capital. NO trade is a
high-quality decision when edge is unclear. Quality over quantity — typically
0–3 good ideas per tick, often zero.

## Pre-trade checklist (ALL must pass)

1. **Technical setup** — start from ranked scanner ideas (`list_intraday_ideas` /
   `list_ideas`). Prefer higher `heat_score`, tier A, and a real news catalyst
   (`has_news_catalyst`). A leveraged ETF tagged only `atr_leader,gapper` with no
   news is low-quality.
2. **News + sentiment (REQUIRED)** — call `get_news` for the symbol; read the
   aggregated `sentiment_score` (−1..+1) and headlines. Don't trade into strong
   contradictory fresh news.
3. **Analyst view (REQUIRED)** — call `get_analyst_view`; use rating / target /
   upside as a directional cross-check. Don't short a Strong-Buy or buy a
   Strong-Sell unless the tape strongly contradicts. **A proposal that skipped
   either news or analyst view is auto-rejected (`insufficient_diligence`).**
4. **Risk/reward** — ≥ 2:1 reward:risk. Stop tied to structure/ATR — not so tight
   it's noise, not so wide it ruins R:R.
5. **Portfolio fit** — diversify across UNCORRELATED themes; deploy idle cash;
   don't pile into one name/theme.

## Enforced caps (respect them)

- **Per-name:** ≤ ~20% of equity (`day_max_position_pct`), position-aware.
- **Per-theme:** ≤ ~35% of equity (`day_theme_max_pct`) — correlated names
  (SOXL/SOXS/NVDL, UCO/SCO/USO) are ONE bet, not diversification.
- **Concurrency:** ≤ 8 positions. **If the book is full, `exit_position` the
  weakest holding (biggest loser / closest to stop) BEFORE adding a better one.**
- **Sizing:** ~1% account risk to stop; margin is allowed (paper).
- **Cooldown:** a just-traded name is hidden for ~10 min to force rotation.

## Stops & exits (live)

- Every entry sets a `stop_price`; the monitor closes the position the moment
  live price breaches the stop **or** a 2R take-profit target, checked every ~30 s.
- Cut losers quickly; let winners work unless the thesis is invalidated. Don't
  average down.
- A catch-all 5% protective stop backfills any held position lacking a plan
  (`DAY_DEFAULT_STOP_PCT`).

## Instrument rules

- LONG → `side='buy'` (stop below entry). SHORT → `side='sell'` (stop above).
- **Bearish on a leveraged/inverse ETF:** BUY its inverse counterpart (bearish
  semis → buy SOXS, not short SOXL). A short of one is auto-converted to a long
  of the inverse with an equivalent % stop.
- Options: `list_option_contracts` then `propose_option` (1–2 contracts) for
  high-conviction directional plays; prefer liquid, tight-spread strikes.

## Time-of-day

- Open: high volatility, fakeouts. Midday: chop. Power hour: trend/reversal.
- **No new entries after 15:30 ET.** All positions auto-flatten by 15:55 ET —
  the agent holds nothing overnight by design.

## Commands — pull ideas / place / exit / monitor

All run inside the scheduler container (it has the MCP env wired).

```bash
# Current ranked ideas + freshness (sp500 universe by default)
docker compose exec -T scheduler python -c "
from src.mcp_clients.short_term import ShortTermClient
st=ShortTermClient(); st.start()
r=st.list_ideas(mode='intraday', tier='A', limit=15)
for x in (r.get('rows') or r.get('ideas') or []):
    print(x.get('ticker'), 'heat', x.get('heat_score'), x.get('direction'), x.get('signal_tags'))
"

# News + analyst view for a symbol (the REQUIRED diligence)
docker compose exec -T scheduler python -c "
from src.mcp_clients.short_term import ShortTermClient
st=ShortTermClient(); st.start(); print(st.get_news('AAPL', days=2))
"

# Preferred way to act: queue a tick and let the agent run the full disciplined
# loop (diligence gate + stops + caps + logging all applied automatically):
./trader tick day
```

Prefer queuing a tick (or using the Streamlit "Tick day now" button) over raw
Alpaca calls — the agent applies every guardrail and logs the reasoning. Only
place/close directly via `LongTermClient` for corrective/manual actions, and
journal why.

## Reject reasons you'll see (and what they mean)

| reason | meaning |
|---|---|
| `insufficient_diligence` | proposed without `get_news` **and** `get_analyst_view` |
| `theme_at_cap:<theme>` | that correlation theme is already at ~35% |
| `max_concurrent_positions` | book full (8) — exit something first |
| `size_rounded_to_zero` | already at the per-name cap, or stop too wide |
| `stop_on_wrong_side` / `stop_equals_entry` | malformed stop |
| `max_gross_exposure` | (only if a gross cap is configured) |

## Rate-limit reality

Effective trade frequency is bounded by the LLM quota. On the GitHub Models free
tier, `429` (rate limit) and `413` (request too big for the fallback) are common
→ ticks come back `no-op`. Enable **compact mode** to keep trading on the
fallback; a paid key removes the ceiling. This is a quota limit, not a bug.
