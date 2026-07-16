"""Generic function-calling loop.

Run: model -> if tool_calls -> execute each via the registry -> append a
`tool` role message per call -> loop. Bounded by `max_steps`. Every step
(messages sent, calls made, results) is captured for the agent trace.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.llm.provider import ChatResult, LLMProvider, ToolCall, ToolSpec

log = logging.getLogger(__name__)

# Read-only, side-effect-free tools. If the model calls one of these AGAIN with
# identical arguments within a single tick, we short-circuit with a compact
# "already fetched" note instead of re-running the (often expensive) handler —
# this stops the model burning its step budget re-fetching the same idea list and
# nudges it toward a decision. NEVER cache tools with side effects (exit_position,
# propose_trade, propose_option).
_READ_ONLY_TOOLS = frozenset({
    "list_intraday_ideas", "get_quote", "get_news", "get_analyst_view",
    "current_positions", "account_snapshot", "list_option_contracts",
})


@dataclass
class ToolHandler:
    spec: ToolSpec
    fn: Callable[..., Any]


@dataclass
class StepTrace:
    step: int
    tool_calls: list[dict] = field(default_factory=list)  # {name, args, result}
    text: str | None = None


@dataclass
class LoopResult:
    final_text: str | None
    steps: list[StepTrace]
    messages: list[dict]


def _synthesize_summary(steps: list[StepTrace], *, reason: str) -> str:
    """Build a short human-readable summary when the model stops without a final
    message (hit the step budget or deadline mid-work). Keeps the reasoning panel
    from showing nothing by describing what the tick actually did."""
    exits: list[str] = []
    proposals: list[str] = []
    other: list[str] = []
    for s in steps:
        for tc in s.tool_calls:
            name = tc.get("name")
            args = tc.get("args") or {}
            if name in ("exit_position",):
                exits.append(str(args.get("symbol", "?")).upper())
            elif name in ("propose_trade", "propose_option"):
                sym = args.get("symbol") or args.get("occ_symbol") or "?"
                side = args.get("side", "buy")
                proposals.append(f"{side} {str(sym).upper()}")
            elif name not in ("list_intraday_ideas", "get_quote", "get_news",
                              "get_analyst_view", "current_positions",
                              "account_snapshot", "list_option_contracts"):
                other.append(str(name))
    parts: list[str] = []
    if exits:
        parts.append("exited " + ", ".join(dict.fromkeys(exits)))
    if proposals:
        parts.append("proposed " + ", ".join(dict.fromkeys(proposals)))
    if other:
        parts.append("actions: " + ", ".join(dict.fromkeys(other)))
    did = "; ".join(parts) if parts else "gathered data but reached no decision"
    return (f"(auto-summary — {reason} before the model wrote a closing note) "
            f"This tick: {did}.")


def run_tool_loop(
    provider: LLMProvider,
    messages: list[dict],
    handlers: list[ToolHandler],
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    max_steps: int = 8,
    deadline_seconds: float | None = None,
) -> LoopResult:
    registry = {h.spec.name: h for h in handlers}
    tools = [h.spec for h in handlers]
    msgs = list(messages)
    steps: list[StepTrace] = []
    _start = time.monotonic()
    # Remembers read-only calls already made this tick (keyed by name+args) so a
    # duplicate returns a cheap note instead of re-running the handler.
    _seen_readonly: dict[str, int] = {}

    for step in range(1, max_steps + 1):
        # Wall-clock budget: a single tick must not run for minutes (a slow model
        # + retries could otherwise overrun the tick cadence and cause the
        # scheduler to skip subsequent ticks). Stop cleanly with what we have.
        if deadline_seconds is not None and (time.monotonic() - _start) > deadline_seconds:
            log.warning("tool loop exceeded deadline (%.0fs) at step %d; stopping early",
                         deadline_seconds, step)
            summary = _synthesize_summary(steps, reason="deadline reached")
            return LoopResult(final_text=summary, steps=steps, messages=msgs)
        result: ChatResult = provider.chat(msgs, tools=tools or None,
                                            temperature=temperature, max_tokens=max_tokens)
        trace = StepTrace(step=step, text=result.text)

        if not result.tool_calls:
            steps.append(trace)
            msgs.append({"role": "assistant", "content": result.text or ""})
            return LoopResult(final_text=result.text, steps=steps, messages=msgs)

        # Echo assistant tool-call message so the API has the call id context.
        msgs.append({
            "role": "assistant",
            "content": result.text or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in result.tool_calls
            ],
        })

        for tc in result.tool_calls:
            handler = registry.get(tc.name)
            if handler is None:
                err = f"unknown tool: {tc.name}"
                trace.tool_calls.append({"name": tc.name, "args": tc.arguments, "error": err})
                msgs.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                             "content": json.dumps({"error": err})})
                continue
            # De-dup: a repeated read-only call with identical args returns a cheap
            # note so the model doesn't waste its step budget re-fetching.
            if tc.name in _READ_ONLY_TOOLS:
                try:
                    key = tc.name + ":" + json.dumps(tc.arguments, sort_keys=True, default=str)
                except Exception:
                    key = tc.name + ":" + str(tc.arguments)
                prior = _seen_readonly.get(key)
                if prior is not None:
                    note = {"note": f"Already called `{tc.name}` with these arguments "
                                    f"at step {prior} this tick — reuse that result. "
                                    f"Proceed to a decision (propose/exit) or finish.",
                            "cached": True}
                    trace.tool_calls.append({"name": tc.name, "args": tc.arguments,
                                              "result": note})
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                 "content": json.dumps(note)})
                    continue
                _seen_readonly[key] = step
            try:
                out = handler.fn(**tc.arguments)
            except Exception as exc:
                log.exception("tool handler raised: %s", tc.name)
                out = {"error": f"{type(exc).__name__}: {exc}"}
            trace.tool_calls.append({"name": tc.name, "args": tc.arguments, "result": out})
            msgs.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                         "content": json.dumps(out, default=str)})

        steps.append(trace)

    # Hit max_steps without a terminal text — synthesize a summary so the
    # reasoning panel isn't blank and it's clear the tick ran out of steps.
    summary = _synthesize_summary(steps, reason="step budget exhausted")
    return LoopResult(final_text=summary, steps=steps, messages=msgs)
