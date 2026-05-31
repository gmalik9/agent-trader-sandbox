import json

import httpx
import pytest

from src.llm.github_models import GitHubModelsProvider
from src.llm.provider import ToolSpec


def _make(transport, token="t"):
    client = httpx.Client(transport=transport)
    return GitHubModelsProvider(token=token, model="openai/gpt-4o-mini",
                                  endpoint="https://example/v1/chat/completions",
                                  client=client)


def test_request_shape_and_auth_header():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        })

    transport = httpx.MockTransport(handler)
    prov = _make(transport, token="abc")
    res = prov.chat([{"role": "user", "content": "hi"}], tools=None)

    assert captured["headers"]["authorization"] == "Bearer abc"
    assert captured["body"]["model"] == "openai/gpt-4o-mini"
    assert captured["body"]["messages"][0]["content"] == "hi"
    assert "tools" not in captured["body"]
    assert res.text == "ok" and res.tool_calls == []


def test_tools_are_serialized_as_openai_functions():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": None,
                                       "tool_calls": [{"id": "c1", "type": "function",
                                                       "function": {"name": "add",
                                                                    "arguments": "{\"a\":1}"}}]}}],
        })

    transport = httpx.MockTransport(handler)
    prov = _make(transport)
    spec = ToolSpec(name="add", description="add ints",
                    json_schema={"type": "object", "properties": {"a": {"type": "integer"}}})
    res = prov.chat([{"role": "user", "content": "x"}], tools=[spec])

    assert captured["body"]["tools"][0]["function"]["name"] == "add"
    assert captured["body"]["tool_choice"] == "auto"
    assert res.tool_calls[0].name == "add"
    assert res.tool_calls[0].arguments == {"a": 1}


def test_raises_on_http_error():
    transport = httpx.MockTransport(lambda r: httpx.Response(401, text="bad token"))
    prov = _make(transport)
    with pytest.raises(RuntimeError):
        prov.chat([{"role": "user", "content": "x"}])
