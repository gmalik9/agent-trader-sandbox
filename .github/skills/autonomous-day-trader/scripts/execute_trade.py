#!/usr/bin/env python3
"""Execute a trade decision Copilot has made — WITHOUT any LLM/PAT call.

This is the "hands" for COPILOT-DRIVEN mode. Copilot reads gather_context.py,
reasons (that's the LLM part — done by Copilot, not the PAT), then calls this to
act. Sizing, caps, inverse-ETF substitution, price sanity, the live stop plan and
audit logging are all applied by the existing engine code — exactly like an
autonomous tick, minus the LLM.

Enter:
    execute_trade.py --enter --symbol AAPL --side buy \
        --entry 150.00 --stop 143.00 [--target 164.00] \
        --thesis "Breakout above VWAP; positive news + Buy rating; 2:1 R:R"

Exit (cut a loser / take profit / free a slot):
    execute_trade.py --exit --symbol AAPL --reason "thesis invalidated"

Run inside the scheduler container. Prints a JSON result. Every action is written
to agent_runs + data/reasoning_log.jsonl (shows in the dashboard) and journaled.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from _trader import build_agent, close  # noqa: E402


def _journal(kind: str, summary: str, detail: str, symbols, run_id=None) -> None:
    """Best-effort journal append (reuse the skill's writer)."""
    try:
        from journal_append import append_entry
        append_entry(kind=kind, summary=summary, detail=detail,
                     tags=["copilot", kind], symbols=symbols, run_id=run_id)
    except Exception:
        pass


def do_enter(agent, args) -> dict:
    from src.agents.day_trader import _Proposal

    is_option = bool(getattr(args, "occ", None))
    if is_option:
        # Route options straight through the options venue (defined-risk plays).
        from src.agents.day_trader import _OptionProposal
        props = [_OptionProposal(occ_symbol=args.occ.upper(), qty=max(1, int(args.qty or 1)),
                                 side="buy" if args.side != "sell" else "sell",
                                 thesis=args.thesis or "")]
        orders = agent._place_options(props)
        rid = agent._record_run(
            status="ok", prompt="copilot", response=args.thesis or "",
            tools_called=[{"step": 1, "text": "copilot-driven option", "tool_calls": []}],
            decisions=orders, error=None, latency_ms=0)
        _journal("trade", f"Copilot option: {args.side} {args.occ}",
                 args.thesis or "", [args.occ.upper()], run_id=rid)
        return {"ok": True, "run_id": rid, "orders": orders}

    if args.entry is None or args.stop is None:
        raise SystemExit("--entry and --stop are required for an equity/ETF entry")
    side = "sell" if str(args.side).lower() in ("sell", "short") else "buy"
    proposals = [_Proposal(symbol=args.symbol.upper(), entry_price=float(args.entry),
                           stop_price=float(args.stop), side=side,
                           thesis=args.thesis or "")]
    if args.target is not None:
        # carry the target through so the plan uses it (else 2R is derived)
        proposals[0].thesis = (proposals[0].thesis + f" | target {args.target}").strip()

    # Same pipeline as an autonomous tick, minus the LLM: inverse-ETF
    # substitution -> reprice to live market -> size against caps -> place +
    # record the live stop plan. Diligence gate is inactive (no _news_checked on
    # this fresh agent) because Copilot did the diligence itself.
    proposals = agent._maybe_substitute_inverse(proposals)
    proposals = agent._reprice_to_market(proposals)
    decisions = agent._validate_and_size(proposals)
    # apply explicit target if provided (on the accepted decision)
    if args.target is not None:
        for d in decisions:
            if d.accepted:
                d.target_price = float(args.target)
    orders = agent._place(decisions)

    dec_dicts = [d.__dict__ for d in decisions]
    rid = agent._record_run(
        status="ok", prompt="copilot", response=args.thesis or "",
        tools_called=[{"step": 1, "text": "copilot-driven entry", "tool_calls": []}],
        decisions=dec_dicts, error=None, latency_ms=0)

    accepted = [d for d in decisions if d.accepted]
    rejected = [(d.symbol, d.reject_reason) for d in decisions if not d.accepted]
    summary = (f"Copilot entry: {side} {args.symbol.upper()} "
               f"({'placed' if accepted else 'REJECTED ' + str(rejected)})")
    _journal("trade", summary, args.thesis or "",
             [d.symbol for d in decisions], run_id=rid)
    return {"ok": bool(accepted), "run_id": rid,
            "orders": orders, "rejected": rejected}


def do_exit(agent, args) -> dict:
    from src.agents.day_trader import _is_option_symbol
    sym = args.symbol.upper()
    if _is_option_symbol(sym):
        raise SystemExit("options can't be closed via the stock endpoint; they expire")
    held = {p.symbol.upper(): p for p in agent.broker.list_positions(agent.sub_account)}
    if sym not in held or abs(held[sym].qty) <= 1e-9:
        return {"ok": False, "error": "not_held", "symbol": sym}
    res = agent.broker.close_position(sym, sub_account=agent.sub_account, percentage=100.0)
    # retire any active plan
    try:
        from src.sandbox import db as dbm
        aid = dbm.get_account_id(agent.conn, agent.sub_account)
        for pl in dbm.get_active_position_plans(agent.conn, aid):
            if str(pl["symbol"]).upper() == sym:
                dbm.close_position_plan(agent.conn, pl["id"], "copilot_exit")
    except Exception:
        pass
    rec = {"symbol": sym, "reason": args.reason or "copilot_exit",
           "status": getattr(res, "status", None), "order_id": getattr(res, "id", None)}
    rid = agent._record_run(
        status="ok", prompt="copilot", response=args.reason or "exit",
        tools_called=[{"step": 1, "text": "copilot-driven exit", "tool_calls": []}],
        decisions=[{"copilot_exit": rec}], error=None, latency_ms=0)
    _journal("trade", f"Copilot exit: {sym}", args.reason or "", [sym], run_id=rid)
    return {"ok": True, "run_id": rid, **rec}


def main() -> None:
    ap = argparse.ArgumentParser(description="Execute a Copilot trade decision (no LLM).")
    ap.add_argument("--enter", action="store_true", help="open a position")
    ap.add_argument("--exit", dest="do_exit", action="store_true", help="close a position")
    ap.add_argument("--symbol", help="ticker (equity/ETF)")
    ap.add_argument("--side", default="buy", help="buy|sell (short)")
    ap.add_argument("--entry", type=float, default=None)
    ap.add_argument("--stop", type=float, default=None)
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--thesis", default="")
    ap.add_argument("--reason", default="")
    ap.add_argument("--occ", default=None, help="OCC option symbol (routes to options venue)")
    ap.add_argument("--qty", type=int, default=1, help="option contracts")
    args = ap.parse_args()

    if not (args.enter or args.do_exit):
        raise SystemExit("pass --enter or --exit")
    if args.do_exit and not args.symbol:
        raise SystemExit("--exit requires --symbol")
    if args.enter and not (args.symbol or args.occ):
        raise SystemExit("--enter requires --symbol (or --occ for an option)")

    agent, clients = build_agent()
    try:
        result = do_exit(agent, args) if args.do_exit else do_enter(agent, args)
    finally:
        close(clients)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
