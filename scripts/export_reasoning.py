"""Export the agents' reasoning to a durable JSONL learning dataset.

Every trade decision (and every no-trade run) is written as one JSON line
pairing the agent's reasoning + cited data sources with the trade it made. This
file is meant to be replayed later to learn which reasoning/data led to good or
bad outcomes.

Run:    python -m scripts.export_reasoning
Output: data/reasoning_log.jsonl  (append-safe; de-duplicated by run_id+symbol)
"""

from __future__ import annotations

import json
from pathlib import Path

from src.analysis import reasoning as R
from src.config import DATA_DIR, db_path
from src.sandbox import db as dbm

OUT_PATH = DATA_DIR / "reasoning_log.jsonl"


def export(out_path: Path = OUT_PATH) -> int:
    conn = dbm.get_conn(db_path())
    dbm.migrate(conn)
    records = R.learning_records(conn)

    # De-dup against what's already written (run_id + symbol key).
    seen: set[tuple] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                seen.add((r.get("run_id"), r.get("symbol")))
            except json.JSONDecodeError:
                continue

    written = 0
    with out_path.open("a") as fh:
        for rec in records:
            key = (rec["run_id"], rec["symbol"])
            if key in seen:
                continue
            fh.write(json.dumps(rec) + "\n")
            seen.add(key)
            written += 1
    return written


def main() -> None:
    n = export()
    print(f"wrote {n} new reasoning record(s) to {OUT_PATH}")


if __name__ == "__main__":
    main()
