"""GitHub Models provider — OpenAI-compatible chat completions.

Endpoint:    https://models.github.ai/inference/chat/completions
Auth:        `Authorization: Bearer ${GITHUB_TOKEN}` (a PAT with Models access)
Default model: `openai/gpt-4o-mini`
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from src.config import get_settings
from src.llm.provider import ChatResult, LLMProvider, ToolCall, ToolSpec

log = logging.getLogger(__name__)

ENDPOINT = "https://models.github.ai/inference/chat/completions"


class RateLimitError(RuntimeError):
    """Raised on HTTP 429 (rate limit) or 413 (input too large for the model's
    tier) so callers can back off or downshift to a bigger-input model."""

# Newer OpenAI families (GPT-5, o-series reasoning models) reject `max_tokens`
# (require `max_completion_tokens`) and only accept the default temperature.
_NEXTGEN_PREFIXES = ("openai/gpt-5", "openai/o1", "openai/o3", "openai/o4")


def _is_nextgen(model: str) -> bool:
    m = (model or "").lower()
    return any(m.startswith(p) for p in _NEXTGEN_PREFIXES)


class GitHubModelsProvider:
    name = "github"

    def __init__(self, token: str | None = None, model: str | None = None,
                 *, endpoint: str = ENDPOINT, timeout: float = 60.0,
                 client: httpx.Client | None = None) -> None:
        s = get_settings()
        self._token = token or s.github_token
        if not self._token:
            raise ValueError("GITHUB_TOKEN not set")
        self.model = model or s.llm_model
        self.endpoint = endpoint
        self._client = client or httpx.Client(timeout=timeout)
        # Rate-limit retry policy (429s that resolve within a few seconds should
        # not stall a tick — retry with backoff, respecting Retry-After).
        self._max_retries = int(getattr(s, "llm_max_retries", 3) or 0)
        self._retry_backoff = float(getattr(s, "llm_retry_backoff", 1.5) or 1.5)
        self._retry_cap = float(getattr(s, "llm_retry_cap", 8.0) or 8.0)

    def _retry_after(self, resp: httpx.Response, attempt: int) -> float:
        """Seconds to wait before the next retry: honor Retry-After if present,
        else exponential backoff, both capped."""
        hdr = resp.headers.get("retry-after") or resp.headers.get("x-ratelimit-reset-after")
        if hdr:
            try:
                return min(float(hdr), self._retry_cap)
            except (TypeError, ValueError):
                pass
        return min(self._retry_backoff * (2 ** attempt), self._retry_cap)

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if _is_nextgen(self.model):
            # GPT-5 / o-series: token budget uses a different key and these
            # reasoning models only support the default temperature. Reasoning
            # tokens count against the budget, so give it generous headroom.
            body["max_completion_tokens"] = max(max_tokens, 4096)
        else:
            body["max_tokens"] = max_tokens
            body["temperature"] = temperature
        if tools:
            body["tools"] = [_tool_to_openai(t) for t in tools]
            body["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        }
        payload = json.dumps(body)

        # Retry 429s with backoff (respecting Retry-After) so transient throttling
        # doesn't stall the tick. After exhausting retries, raise RateLimitError
        # so the adaptive provider can downshift.
        attempt = 0
        while True:
            resp = self._client.post(self.endpoint, headers=headers, content=payload)
            if resp.status_code == 429:
                if attempt < self._max_retries:
                    wait = self._retry_after(resp, attempt)
                    log.warning("github models 429 (%s); retry %d/%d in %.1fs",
                                 self.model, attempt + 1, self._max_retries, wait)
                    time.sleep(wait)
                    attempt += 1
                    continue
                raise RateLimitError(f"github models 429 (rate limited): {resp.text[:200]}")
            if resp.status_code == 413:
                # Input too large for this model's tier (e.g. gpt-5-mini's 4k cap).
                # Treat like a rate-limit so callers downshift to a bigger-input model.
                raise RateLimitError(f"github models 413 (tokens_limit_reached): {resp.text[:200]}")
            if resp.status_code >= 400:
                raise RuntimeError(f"github models {resp.status_code}: {resp.text[:500]}")
            return _parse_openai_response(resp.json())


def _tool_to_openai(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.json_schema,
        },
    }


def _parse_openai_response(data: dict[str, Any]) -> ChatResult:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    text = msg.get("content")
    raw_tool_calls = msg.get("tool_calls") or []
    tool_calls: list[ToolCall] = []
    for tc in raw_tool_calls:
        fn = tc.get("function", {})
        args_raw = fn.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except json.JSONDecodeError:
            args = {"_raw": args_raw}
        tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args))
    return ChatResult(text=text, tool_calls=tool_calls, raw=data)
