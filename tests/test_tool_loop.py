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
    # Hitting max_steps now yields a synthesized summary (never a blank reasoning).
    assert res.final_text is not None
    assert "auto-summary" in res.final_text
    assert "step budget exhausted" in res.final_text


def test_deadline_yields_summary_not_blank():
    prov = _SlowLoopingProvider(delay=0.05)
    res = run_tool_loop(prov, [{"role": "user", "content": "go"}], _noop_handler(),
                        max_steps=100, deadline_seconds=0.15)
    assert res.final_text is not None
    assert "deadline reached" in res.final_text


class _ScriptedProvider:
    """Returns queued ChatResults in order."""

    name = "scripted"
    model = "scripted"

    def __init__(self, results):
        self._q = list(results)
        self.calls = 0

    def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=1024):
        self.calls += 1
        return self._q.pop(0)


def test_readonly_tool_deduplicated_within_tick():
    """A repeated read-only call with identical args returns a cached note and
    does NOT re-run the handler."""
    run_count = {"n": 0}

    def _ideas(**kw):
        run_count["n"] += 1
        return {"ideas": [{"ticker": "AAA"}]}

    handlers = [ToolHandler(spec=ToolSpec(name="list_intraday_ideas", description="ideas",
                                           json_schema={"type": "object"}), fn=_ideas)]
    prov = _ScriptedProvider([
        ChatResult(text=None, tool_calls=[ToolCall(id="c1", name="list_intraday_ideas",
                                                    arguments={"tier": "A", "limit": 6})]),
        ChatResult(text=None, tool_calls=[ToolCall(id="c2", name="list_intraday_ideas",
                                                    arguments={"tier": "A", "limit": 6})]),
        ChatResult(text="done", tool_calls=[]),
    ])
    res = run_tool_loop(prov, [{"role": "user", "content": "go"}], handlers,
                        max_steps=8, deadline_seconds=None)
    assert res.final_text == "done"
    assert run_count["n"] == 1          # handler ran once, second call was cached
    # The second step's trace shows a cached result.
    cached = [tc for s in res.steps for tc in s.tool_calls
              if tc.get("result", {}).get("cached")]
    assert len(cached) == 1


def test_side_effect_tool_not_deduplicated():
    """A tool with side effects (propose_trade) is NOT cached — repeated calls run."""
    run_count = {"n": 0}

    def _propose(**kw):
        run_count["n"] += 1
        return {"ok": True, "buffered": run_count["n"]}

    handlers = [ToolHandler(spec=ToolSpec(name="propose_trade", description="p",
                                           json_schema={"type": "object"}), fn=_propose)]
    prov = _ScriptedProvider([
        ChatResult(text=None, tool_calls=[ToolCall(id="c1", name="propose_trade",
                                                    arguments={"symbol": "AAA"})]),
        ChatResult(text=None, tool_calls=[ToolCall(id="c2", name="propose_trade",
                                                    arguments={"symbol": "AAA"})]),
        ChatResult(text="done", tool_calls=[]),
    ])
    run_tool_loop(prov, [{"role": "user", "content": "go"}], handlers,
                  max_steps=8, deadline_seconds=None)
    assert run_count["n"] == 2          # both proposals executed (not cached)

