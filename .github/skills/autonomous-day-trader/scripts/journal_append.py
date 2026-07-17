#!/usr/bin/env python3
"""Append one structured entry to the trader's self-learning journal.

The journal (`data/trader_journal.jsonl`) is append-only, one JSON object per
line. It is the human-like memory: observations, actions, trades and lessons that
you review and distil into `data/strategy_notes.md`.

Examples:
    python journal_append.py --kind session_open \
        --summary "Down-tape open; plan to fade semis only with a catalyst" \
        --tags plan,semis

    python journal_append.py --kind observation \
        --summary "Idea pool 60% semis shorts; book crowding one-way" \
        --detail "AMAT/MU/MRVL tier-A. Watching power-hour reversal." \
        --tags semis,regime --symbols AMAT,MU,MRVL --run-id 5461

    python journal_append.py --kind lesson \
        --summary "5% stop too tight for high-ATR names" --tags risk,stops

Run inside the scheduler container so it writes to the mounted data/ volume:
    docker compose exec -T scheduler python \
        .github/skills/autonomous-day-trader/scripts/journal_append.py --kind ... --summary ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make `import src.*` work when this script is run by path.
_APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

VALID_KINDS = {
    "session_open", "observation", "action", "trade", "lesson", "session_close",
}


def _journal_path() -> Path:
    try:
        from src.config import DATA_DIR
        base = Path(DATA_DIR)
    except Exception:
        base = Path("data")
    base.mkdir(parents=True, exist_ok=True)
    return base / "trader_journal.jsonl"


def append_entry(*, kind: str, summary: str, detail: str = "",
                 tags: list[str] | None = None, symbols: list[str] | None = None,
                 run_id: int | None = None, path: Path | None = None) -> dict:
    if kind not in VALID_KINDS:
        raise SystemExit(f"--kind must be one of {sorted(VALID_KINDS)} (got {kind!r})")
    if not summary or not summary.strip():
        raise SystemExit("--summary is required and cannot be empty")
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        "summary": summary.strip(),
        "detail": (detail or "").strip(),
        "tags": [t.strip() for t in (tags or []) if t.strip()],
        "refs": {},
    }
    if symbols:
        entry["refs"]["symbols"] = [s.strip().upper() for s in symbols if s.strip()]
    if run_id is not None:
        entry["refs"]["run_id"] = run_id
    p = path or _journal_path()
    with p.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def _csv(v: str | None) -> list[str]:
    return [x for x in (v or "").split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Append a journal entry.")
    ap.add_argument("--kind", required=True, help=f"one of {sorted(VALID_KINDS)}")
    ap.add_argument("--summary", required=True, help="one-line takeaway")
    ap.add_argument("--detail", default="", help="optional longer note")
    ap.add_argument("--tags", default="", help="comma-separated tags")
    ap.add_argument("--symbols", default="", help="comma-separated tickers")
    ap.add_argument("--run-id", type=int, default=None, help="related agent_runs.id")
    args = ap.parse_args()

    entry = append_entry(
        kind=args.kind, summary=args.summary, detail=args.detail,
        tags=_csv(args.tags), symbols=_csv(args.symbols), run_id=args.run_id,
    )
    print(f"journaled [{entry['kind']}] -> {_journal_path()}")
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
