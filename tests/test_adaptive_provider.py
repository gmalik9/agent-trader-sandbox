"""Tests for the adaptive provider's throttle-downshift + recovery."""

from __future__ import annotations

import time

from src.llm.adaptive import AdaptiveGitHubProvider
from src.llm.github_models import RateLimitError
from src.llm.provider import ChatResult


class FakeModel:
    def __init__(self, name, *, rate_limited=False):
        self.name = name
        self.rate_limited = rate_limited
        self.calls = 0

    def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=1024):
        self.calls += 1
        if self.rate_limited:
            raise RateLimitError("429")
        return ChatResult(text=self.name, tool_calls=[])


def _provider(primary, fallback, cooldown=100.0):
    p = AdaptiveGitHubProvider.__new__(AdaptiveGitHubProvider)
    p._primary = primary
    p._fallback = fallback
    p.primary_model = "openai/gpt-5"
    p.fallback_model = "openai/gpt-5-mini"
    p.cooldown_seconds = cooldown
    p._downshifted_until = 0.0
    return p


def test_uses_primary_normally():
    prim, fb = FakeModel("primary"), FakeModel("fallback")
    p = _provider(prim, fb)
    res = p.chat([{"role": "user", "content": "hi"}])
    assert res.text == "primary"
    assert p.model == "openai/gpt-5"
    assert fb.calls == 0


def test_downshifts_on_rate_limit_and_retries_fallback():
    prim = FakeModel("primary", rate_limited=True)
    fb = FakeModel("fallback")
    p = _provider(prim, fb)
    res = p.chat([{"role": "user", "content": "hi"}])
    assert res.text == "fallback"          # transparently retried on the cheaper model
    assert p.model == "openai/gpt-5-mini"  # now in cooldown
    # Subsequent call within cooldown goes straight to fallback (primary untouched).
    prim.calls = 0
    p.chat([{"role": "user", "content": "again"}])
    assert prim.calls == 0
    assert fb.calls == 2


def test_recovers_to_primary_after_cooldown():
    prim = FakeModel("primary", rate_limited=True)
    fb = FakeModel("fallback")
    p = _provider(prim, fb, cooldown=0.05)
    p.chat([{"role": "user", "content": "hi"}])   # trip the downshift
    assert p.model == "openai/gpt-5-mini"
    time.sleep(0.06)                               # let cooldown lapse
    prim.rate_limited = False                      # primary healthy again
    res = p.chat([{"role": "user", "content": "back"}])
    assert res.text == "primary"
    assert p.model == "openai/gpt-5"
