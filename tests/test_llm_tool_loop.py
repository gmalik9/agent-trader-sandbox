from __future__ import annotations

from src.llm.provider import ChatResult, ToolCall, ToolSpec
from src.llm.tool_loop import ToolHandler, run_tool_loop


class ScriptedProvider:
    """Replay a queue of ChatResult objects in order."""

    name = "scripted"
    model = "scripted"

    def __init__(self, results):
        self._queue = list(results)
        self.last_messages = None

    def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=1024):
        self.last_messages = messages
        return self._queue.pop(0)


def test_text_only_returns_immediately():
    prov = ScriptedProvider([ChatResult(text="hello world")])
    out = run_tool_loop(prov, [{"role": "user", "content": "hi"}], handlers=[])
    assert out.final_text == "hello world"
    assert len(out.steps) == 1
    assert out.steps[0].tool_calls == []


def test_tool_call_executes_and_continues():
    calls = []

    def add(a: int, b: int) -> dict:
        calls.append((a, b))
        return {"sum": a + b}

    spec = ToolSpec(name="add", description="add two ints", json_schema={
        "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    })
    prov = ScriptedProvider([
        ChatResult(text=None, tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]),
        ChatResult(text="five"),
    ])
    out = run_tool_loop(prov, [{"role": "user", "content": "add"}],
                        handlers=[ToolHandler(spec=spec, fn=add)])

    assert out.final_text == "five"
    assert calls == [(2, 3)]
    assert len(out.steps) == 2
    assert out.steps[0].tool_calls[0]["result"] == {"sum": 5}


def test_unknown_tool_is_reported_to_model():
    prov = ScriptedProvider([
        ChatResult(text=None, tool_calls=[ToolCall(id="c1", name="nope", arguments={})]),
        ChatResult(text="ok i wont"),
    ])
    out = run_tool_loop(prov, [{"role": "user", "content": "x"}], handlers=[])
    assert out.final_text == "ok i wont"
    assert out.steps[0].tool_calls[0]["error"].startswith("unknown tool")


def test_tool_exception_is_captured_not_raised():
    def bomb(**_):
        raise RuntimeError("kaboom")

    spec = ToolSpec(name="bomb", description="explodes", json_schema={"type": "object"})
    prov = ScriptedProvider([
        ChatResult(text=None, tool_calls=[ToolCall(id="c1", name="bomb", arguments={})]),
        ChatResult(text="moving on"),
    ])
    out = run_tool_loop(prov, [{"role": "user", "content": "x"}],
                        handlers=[ToolHandler(spec=spec, fn=bomb)])
    assert out.final_text == "moving on"
    assert "RuntimeError" in out.steps[0].tool_calls[0]["result"]["error"]


def test_max_steps_cap_returns_summary_text():
    # Model keeps calling forever.
    spec = ToolSpec(name="ping", description="", json_schema={"type": "object"})
    prov = ScriptedProvider([
        ChatResult(text=None, tool_calls=[ToolCall(id=f"c{i}", name="ping", arguments={})])
        for i in range(5)
    ])
    out = run_tool_loop(prov, [{"role": "user", "content": "x"}],
                        handlers=[ToolHandler(spec=spec, fn=lambda: {"pong": True})],
                        max_steps=3)
    # Hitting max_steps now returns a synthesized summary (never a blank reasoning).
    assert out.final_text is not None and "auto-summary" in out.final_text
    assert len(out.steps) == 3
