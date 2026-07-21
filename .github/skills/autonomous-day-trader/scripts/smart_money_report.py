#!/usr/bin/env python3
"""Smart-money report: insider (C-suite) + political (Congress) trade activity.

Two modes, both LLM-free (pure data pull — safe to run any time):

  # Which stocks have unusual insider/political activity right now?
  docker compose exec -T scheduler python \
      .github/skills/autonomous-day-trader/scripts/smart_money_report.py --market

  # Deep-dive one symbol's insider + political disclosures
  docker compose exec -T scheduler python \
      .github/skills/autonomous-day-trader/scripts/smart_money_report.py --symbol NVDA

  # Approximate net positions a given person built across the market
  # (individual stocks + sectors), reconstructed from disclosed transactions
  docker compose exec -T scheduler python \
      .github/skills/autonomous-day-trader/scripts/smart_money_report.py --person "Pelosi"

Requires FMP_API_KEY (insider + political) or FINNHUB_API_KEY (insider only) in
secrets.toml / env. Prints JSON. Feed it to Copilot as extra context.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)


def _client():
    from src.config import get_settings
    from src.smart_money import SmartMoneyClient

    s = get_settings()
    return SmartMoneyClient(
        fmp_api_key=getattr(s, "fmp_api_key", "") or "",
        finnhub_api_key=getattr(s, "finnhub_api_key", "") or "",
        lookback_days=int(getattr(s, "smart_money_lookback_days", 90) or 90),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--market", action="store_true",
                   help="rank stocks by net disclosed insider/political flow")
    g.add_argument("--symbol", help="deep-dive one symbol")
    g.add_argument("--person", help="approximate a person's positions market-wide")
    ap.add_argument("--limit", type=int, default=25)
    args = ap.parse_args()

    sm = _client()
    try:
        if not sm.available:
            print(json.dumps({"available": False,
                              "reason": "set FMP_API_KEY (insider+political) or "
                                        "FINNHUB_API_KEY (insider only)"}, indent=2))
            return
        if args.market:
            out = sm.market_activity(limit=args.limit)
        elif args.symbol:
            out = sm.symbol_activity(args.symbol)
        else:
            out = sm.person_positions(args.person, limit=args.limit)
        print(json.dumps(out, indent=2, default=str))
    finally:
        sm.close()


if __name__ == "__main__":
    main()
