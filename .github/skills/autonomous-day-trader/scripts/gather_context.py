#!/usr/bin/env python3
"""Gather everything Copilot needs to make a trading decision — in ONE JSON dump.

This is the "eyes" for COPILOT-DRIVEN mode: instead of the scheduler calling an
LLM (which burns the GitHub Models PAT), Copilot itself reads this context each
minute, reasons about it, and then acts via execute_trade.py. No LLM/API call is
made here — it only reads the scanner, news, analyst and broker data.

    docker compose exec -T scheduler python \
        .github/skills/autonomous-day-trader/scripts/gather_context.py [--ideas N] [--news M]

Output: a JSON object with market state, account, current book (with per-holding
P&L + distance-to-stop), and the top ranked ideas each enriched with news
sentiment + analyst view (the REQUIRED diligence). Feed it to Copilot to decide.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from _trader import build_agent, close  # noqa: E402

try:
    from src.sandbox.clock import is_market_open as _is_market_open
except Exception:  # pragma: no cover
    _is_market_open = None


def _theme_of(sym: str):
    try:
        from src.agents.day_trader import _theme_of as t
        return t(sym)
    except Exception:
        return sym


def build_context(n_ideas: int, n_news: int) -> dict:
    agent, clients = build_agent()
    try:
        now = datetime.now(timezone.utc)
        out: dict = {
            "ts": now.isoformat(timespec="seconds"),
            "market_open": _is_market_open(now) if _is_market_open else None,
        }

        # --- account + book (Alpaca leg = source of truth) ---
        acct = agent.broker.get_account(agent.sub_account)
        equity = float(acct.equity or 0)
        out["account"] = {
            "equity": round(equity, 2), "cash": round(float(acct.cash or 0), 2),
            "positions_value": round(float(acct.positions_value or 0), 2),
        }

        from src.config import get_settings
        s = get_settings()
        cap_pct = float(getattr(s, "day_max_position_pct", 0.20) or 0.20)
        theme_cap = float(getattr(s, "day_theme_max_pct", 0.35) or 0.35)
        max_pos = 8
        try:
            from src.agents.day_trader import MAX_CONCURRENT_POSITIONS
            max_pos = MAX_CONCURRENT_POSITIONS
        except Exception:
            pass

        # active stop plans keyed by symbol (for distance-to-stop)
        plans = {}
        try:
            from src.sandbox import db as dbm
            aid = dbm.get_account_id(agent.conn, agent.sub_account)
            for pl in dbm.get_active_position_plans(agent.conn, aid):
                plans[str(pl["symbol"]).upper()] = dict(pl)
        except Exception:
            pass

        holdings = []
        theme_exposure: dict[str, float] = {}
        for p in agent.broker.list_positions(agent.sub_account):
            q = p.qty
            if abs(q) <= 1e-9 or equity <= 0:
                continue
            sym = p.symbol.upper()
            mark = p.mark_price or 0.0
            pct = 100.0 * abs(q) * mark / equity
            th = _theme_of(sym)
            theme_exposure[th] = theme_exposure.get(th, 0.0) + pct
            is_long = q > 0
            stop = plans.get(sym, {}).get("stop_price")
            d2s = None
            if stop and mark > 0:
                d2s = round((100.0 * (mark - stop) / mark) if is_long
                            else (100.0 * (stop - mark) / mark), 2)
            holdings.append({
                "symbol": sym, "side": "long" if is_long else "short",
                "qty": q, "pct_of_equity": round(pct, 1),
                "unrealized_pnl": round(getattr(p, "unrealized_pnl", 0.0) or 0.0, 2),
                "avg_cost": round(p.avg_cost or 0.0, 4), "mark": round(mark, 4),
                "stop": round(stop, 4) if stop else None, "pct_to_stop": d2s,
                "theme": th,
            })
        holdings.sort(key=lambda h: (h["unrealized_pnl"],
                                     h["pct_to_stop"] if h["pct_to_stop"] is not None else 1e9))
        out["book"] = {
            "holdings": holdings,
            "open_position_count": len(holdings),
            "max_positions": max_pos,
            "book_full": len(holdings) >= max_pos,
            "per_name_cap_pct": round(cap_pct * 100, 1),
            "per_theme_cap_pct": round(theme_cap * 100, 1),
            "theme_exposure_pct": {k: round(v, 1) for k, v in theme_exposure.items()},
            "on_cooldown": sorted(agent._recent_traded_symbols()),
        }

        # --- ranked ideas + REQUIRED diligence (news + analyst) per idea ---
        held_syms = {h["symbol"] for h in holdings}
        res = agent.short_term.list_ideas(mode="intraday", tier="A", limit=max(n_ideas, 10))
        rows = (res.get("rows") or res.get("ideas") or []) if isinstance(res, dict) else []
        ideas = []
        for r in rows[:n_ideas]:
            sym = str(r.get("ticker") or r.get("symbol") or "").upper()
            if not sym:
                continue
            idea = {
                "ticker": sym, "direction": r.get("direction"), "tier": r.get("tier"),
                "heat_score": r.get("heat_score"),
                "signal_tags": r.get("signal_tags"),
                "has_news_catalyst": "news_spike" in str(r.get("signal_tags") or ""),
                "entry": r.get("entry"), "stop": r.get("stop"),
                "target": r.get("target"), "rr": r.get("rr"),
                "theme": _theme_of(sym), "already_held": sym in held_syms,
            }
            if n_news > 0:
                try:
                    news = agent._news(sym, days=2)
                    idea["news"] = {k: news.get(k) for k in
                                    ("sentiment_score", "sentiment_label",
                                     "bullish", "bearish") if k in news}
                    heads = news.get("headlines") or news.get("top_headlines") or []
                    idea["news"]["headlines"] = heads[:3]
                except Exception as e:  # noqa: BLE001
                    idea["news"] = {"error": str(e)[:80]}
                try:
                    av = agent._analyst_view(sym)
                    idea["analyst"] = {k: av.get(k) for k in
                                       ("rating", "target_price", "upside_pct",
                                        "analyst_count", "sentiment") if k in av}
                except Exception as e:  # noqa: BLE001
                    idea["analyst"] = {"error": str(e)[:80]}
            ideas.append(idea)
        out["ideas"] = ideas
        out["guidance"] = (
            "COPILOT-DRIVEN: you are the trader. Pick 0-3 best setups. REQUIRE a "
            "news + analyst cross-check (already included per idea). Enter only with "
            "a defined stop and >=2:1 R:R. If book_full, exit the weakest holding "
            "first. Then call execute_trade.py to act. No new entries after 15:30 ET."
        )
        return out
    finally:
        close(clients)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ideas", type=int, default=6, help="number of top ideas")
    ap.add_argument("--news", type=int, default=6,
                    help="enrich the top N ideas with news+analyst (0 to skip)")
    args = ap.parse_args()
    ctx = build_context(args.ideas, min(args.news, args.ideas))
    print(json.dumps(ctx, indent=2, default=str))


if __name__ == "__main__":
    main()
