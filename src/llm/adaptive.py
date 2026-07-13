"""Adaptive LLM provider — run the strongest model, downshift on throttling.

Strategy (per the trading objective: run at 1-minute cadence on the best model,
but never stall when the model is rate-limited):

- Normally use the **primary** model (e.g. ``openai/gpt-5``).
- On a rate-limit (HTTP 429) from the primary, transparently retry the *same*
  request on a cheaper **fallback** model (e.g. ``openai/gpt-5-mini``) and enter
  a cooldown during which all calls go to the fallback.
- After the cooldown elapses, probe the primary again; if it succeeds we recover
  to the superior model automatically.

This keeps the agent trading every tick even while the top model is throttled,
then upgrades back once quota frees up.
"""

from __future__ import annotations

import logging
import time

from src.llm.github_models import GitHubModelsProvider, RateLimitError
from src.llm.provider import ChatResult, ToolSpec

log = logging.getLogger(__name__)


class AdaptiveGitHubProvider:
    name = "github-adaptive"

    def __init__(self, *, primary_model: str = "openai/gpt-5",
                 fallback_model: str = "openai/gpt-5-mini",
                 cooldown_seconds: float = 120.0,
                 token: str | None = None) -> None:
        self._primary = GitHubModelsProvider(token=token, model=primary_model)
        self._fallback = GitHubModelsProvider(token=token, model=fallback_model)
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.cooldown_seconds = cooldown_seconds
        self._downshifted_until = 0.0

    @property
    def model(self) -> str:
        return self.fallback_model if self._is_downshifted() else self.primary_model

    def _is_downshifted(self) -> bool:
        return time.monotonic() < self._downshifted_until

    def chat(self, messages, *, tools: list[ToolSpec] | None = None,
             temperature: float = 0.2, max_tokens: int = 1024) -> ChatResult:
        # If we're in a cooldown, use the fallback directly.
        if self._is_downshifted():
            try:
                return self._fallback.chat(messages, tools=tools,
                                            temperature=temperature, max_tokens=max_tokens)
            except RateLimitError:
                # Even the fallback is throttled — extend cooldown and re-raise.
                self._downshifted_until = time.monotonic() + self.cooldown_seconds
                raise

        # Normal path: try the primary; on 429, downshift and retry once.
        try:
            return self._primary.chat(messages, tools=tools,
                                        temperature=temperature, max_tokens=max_tokens)
        except RateLimitError:
            self._downshifted_until = time.monotonic() + self.cooldown_seconds
            log.warning("primary model %s rate-limited; downshifting to %s for %.0fs",
                         self.primary_model, self.fallback_model, self.cooldown_seconds)
            return self._fallback.chat(messages, tools=tools,
                                        temperature=temperature, max_tokens=max_tokens)
