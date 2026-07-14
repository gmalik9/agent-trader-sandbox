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

    for step in range(1, max_steps + 1):
        # Wall-clock budget: a single tick must not run for minutes (a slow model
        # + retries could otherwise overrun the tick cadence and cause the
        # scheduler to skip subsequent ticks). Stop cleanly with what we have.
        if deadline_seconds is not None and (time.monotonic() - _start) > deadline_seconds:
            log.warning("tool loop exceeded deadline (%.0fs) at step %d; stopping early",
                         deadline_seconds, step)
            break
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
            try:
                out = handler.fn(**tc.arguments)
            except Exception as exc:
                log.exception("tool handler raised: %s", tc.name)
                out = {"error": f"{type(exc).__name__}: {exc}"}
            trace.tool_calls.append({"name": tc.name, "args": tc.arguments, "result": out})
            msgs.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                         "content": json.dumps(out, default=str)})

        steps.append(trace)

    # Hit max_steps without a terminal text.
    return LoopResult(final_text=None, steps=steps, messages=msgs)
