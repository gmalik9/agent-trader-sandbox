"""LLM provider protocol + shared dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolSpec:
    name: str
    description: str
    json_schema: dict[str, Any]  # JSON Schema for the tool's arguments


@dataclass
class ToolCall:
    id: str            # provider-assigned call id (echoed back in the tool message)
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    name: str
    model: str

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ChatResult: ...
