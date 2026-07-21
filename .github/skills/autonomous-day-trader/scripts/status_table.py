#!/usr/bin/env python3
"""Render markdown tables of the book + recent activity — the per-phase report.

Every pass of the Copilot-driven loop should show WHERE THE BOOK STANDS and WHAT
JUST HAPPENED, as two compact tables:

  1. Holdings   — symbol, side, qty, avg, mark, unrealized P&L, stop, %-to-stop,
                  %-of-equity, theme (sorted worst-P&L / closest-to-stop first).
  2. Recent activity — the last N journal entries (time, kind, summary, symbols).

Read-only. Bound to the skill's dedicated paper account (via `_trader`), so it
reports the same account the skill trades. Run inside the scheduler container:

    docker compose exec -T scheduler python \
        .github/skills/autonomous-day-trader/scripts/status_table.py [--activity N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Running a script by path puts the script's dir on sys.path, not the app root —
# add the repo root (4 levels up) so `import src.*` works.
_APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)


def _fmt(v, nd: int = 2, dash: str = "—") -> str:
    if v is None:
        return dash
    try:
        return f"{float(v):,.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Return a GitHub-flavoured markdown table (or a placeholder if empty)."""
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    if not rows:
        empty = "| " + " | ".join(["_(none)_"] + [""] * (len(headers) - 1)) + " |"
        return "\n".join([line, sep, empty])
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line, sep, *body])


def holdings_table(book: dict) -> str:
    holdings = book.get("holdings") or []
    headers = ["Symbol", "Side", "Qty", "Avg", "Mark", "P&L $",
               "Stop", "%→Stop", "%Eq", "Theme"]
    rows: list[list[str]] = []
    for h in holdings:
        pnl = h.get("unrealized_pnl")
        pnl_s = ("+" if (pnl or 0) >= 0 else "") + _fmt(pnl)
        rows.append([
            str(h.get("symbol", "")),
            str(h.get("side", "")),
            _fmt(h.get("qty"), 0),
            _fmt(h.get("avg_cost"), 2),
            _fmt(h.get("mark"), 2),
            pnl_s,
            _fmt(h.get("stop"), 2),
            _fmt(h.get("pct_to_stop"), 2),
            _fmt(h.get("pct_of_equity"), 1),
            str(h.get("theme", "")),
        ])
    return _md_table(headers, rows)


def _journal_path() -> Path:
    try:
        from src.config import DATA_DIR
        base = Path(DATA_DIR)
    except Exception:
        base = Path("data")
    return base / "trader_journal.jsonl"


def recent_activity_table(n: int) -> str:
    headers = ["Time (UTC)", "Kind", "Summary", "Symbols"]
    rows: list[list[str]] = []
    p = _journal_path()
    if p.exists():
        try:
            lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        except Exception:
            lines = []
        for ln in reversed(lines[-n:]):
            try:
                e = json.loads(ln)
            except Exception:
                continue
            ts = str(e.get("ts", ""))
            hhmmss = ts[11:19] if len(ts) >= 19 else ts
            summ = (e.get("summary") or "").replace("|", "\\|")
            if len(summ) > 70:
                summ = summ[:67] + "…"
            syms = ",".join((e.get("refs") or {}).get("symbols") or [])
            rows.append([hhmmss, str(e.get("kind", "")), summ, syms])
    return _md_table(headers, rows)


def render(activity_n: int) -> str:
    # Reuse the exact book computation the loop already relies on (account,
    # holdings with P&L + distance-to-stop). n_ideas=0 keeps it fast/read-only.
    from gather_context import build_context

    ctx = build_context(0, 0)
    acc = ctx.get("account", {})
    book = ctx.get("book", {})
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")

    out: list[str] = []
    out.append(f"### Book & activity @ {now} UTC")
    out.append(
        f"account **PA3WWRZG806F** · equity {_fmt(acc.get('equity'))} · "
        f"cash {_fmt(acc.get('cash'))} · positions {book.get('open_position_count', 0)}"
        f"/{book.get('max_positions', 8)}"
        f"{' · BOOK FULL' if book.get('book_full') else ''} · "
        f"market_open={ctx.get('market_open')}"
    )
    out.append("")
    out.append("**Holdings**")
    out.append(holdings_table(book))
    out.append("")
    out.append(f"**Recent activity** (last {activity_n})")
    out.append(recent_activity_table(activity_n))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Markdown holdings + recent-activity tables.")
    ap.add_argument("--activity", type=int, default=8,
                    help="how many recent journal entries to show")
    args = ap.parse_args()
    print(render(args.activity))


if __name__ == "__main__":
    main()
