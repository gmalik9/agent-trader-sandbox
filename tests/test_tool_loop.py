"""Tool-loop tests — deadline budget + basic termination."""

from __future__ import annotations

import time

from src.llm.provider import ChatResult, ToolCall, ToolSpec
from src.llm.tool_loop import ToolHandler, run_tool_loop


class _SlowLoopingProvider:
    """Always returns a tool call, sleeping `delay` seconds per chat() call —
    simulates a slow model that would otherwise loop to max_steps."""

    name = "slow"
    model = "slow"

    def __init__(self, delay: float):
        self.delay = delay
        self.calls = 0

    def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=1024):
        self.calls += 1
        time.sleep(self.delay)
        return ChatResult(text=None, tool_calls=[ToolCall(
            id=f"c{self.calls}", name="noop", arguments={})])


def _noop_handler():
    return [ToolHandler(spec=ToolSpec(name="noop", description="noop",
                                       json_schema={"type": "object"}),
                        fn=lambda **kw: {"ok": True})]


def test_deadline_stops_loop_early():
    prov = _SlowLoopingProvider(delay=0.05)
    res = run_tool_loop(prov, [{"role": "user", "content": "go"}], _noop_handler(),
                        max_steps=100, deadline_seconds=0.15)
    # Without the deadline this would run 100 steps; the budget caps it to a few.
    assert prov.calls < 100
    assert prov.calls <= 5


def test_no_deadline_runs_to_max_steps():
    prov = _SlowLoopingProvider(delay=0.0)
    res = run_tool_loop(prov, [{"role": "user", "content": "go"}], _noop_handler(),
                        max_steps=4, deadline_seconds=None)
    assert prov.calls == 4
    assert res.final_text is None       # hit max_steps without terminal text
