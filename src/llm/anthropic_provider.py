"""Anthropic provider — minimal adapter to the Messages API.

Translates OpenAI-style messages + tools into Anthropic's schema and back.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from src.config import get_settings
from src.llm.provider import ChatResult, ToolCall, ToolSpec

log = logging.getLogger(__name__)

ENDPOINT = "https://api.anthropic.com/v1/messages"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 *, endpoint: str = ENDPOINT, timeout: float = 60.0,
                 client: httpx.Client | None = None) -> None:
        s = get_settings()
        key = api_key or s.anthropic_api_key
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self._key = key
        self.model = model or "claude-3-5-sonnet-latest"
        self.endpoint = endpoint
        self._client = client or httpx.Client(timeout=timeout)

    def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=1024):
        system = ""
        cleaned = []
        for m in messages:
            if m.get("role") == "system":
                system += (m.get("content") or "") + "\n"
                continue
            cleaned.append(_to_anthropic_message(m))

        body: dict[str, Any] = {
            "model": self.model,
            "messages": cleaned,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            body["system"] = system.strip()
        if tools:
            body["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.json_schema}
                for t in tools
            ]

        resp = self._client.post(
            self.endpoint,
            headers={
                "x-api-key": self._key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            content=json.dumps(body),
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"anthropic {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        return _parse_anthropic_response(data)


def _to_anthropic_message(m: dict) -> dict:
    role = m["role"]
    if role == "tool":
        # OpenAI tool message → Anthropic tool_result content block
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": m.get("content", ""),
            }],
        }
    return {"role": role, "content": m.get("content", "")}


def _parse_anthropic_response(data: dict) -> ChatResult:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in data.get("content", []) or []:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(ToolCall(
                id=block.get("id", ""),
                name=block.get("name", ""),
                arguments=block.get("input", {}) or {},
            ))
    return ChatResult(text="\n".join(text_parts) if text_parts else None,
                       tool_calls=tool_calls, raw=data)
