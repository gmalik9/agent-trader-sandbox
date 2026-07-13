"""Pick an LLM provider based on settings."""

from __future__ import annotations

from src.config import get_settings
from src.llm.provider import LLMProvider


def get_provider(name: str | None = None) -> LLMProvider:
    s = get_settings()
    chosen = (name or s.llm_provider or "github").lower()
    if chosen == "github":
        # For the strongest models (gpt-5 / o-series) use the adaptive provider
        # so a rate-limited primary transparently downshifts to a cheaper model
        # and recovers automatically — keeps the agent trading every tick.
        model = (s.llm_model or "").lower()
        if model.startswith("openai/gpt-5") or model.startswith("openai/o"):
            from src.llm.adaptive import AdaptiveGitHubProvider
            fallback = "openai/gpt-5-mini" if model.startswith("openai/gpt-5") else "openai/gpt-4o-mini"
            if model in ("openai/gpt-5-mini", "openai/gpt-5-nano"):
                fallback = "openai/gpt-4o-mini"
            return AdaptiveGitHubProvider(primary_model=s.llm_model, fallback_model=fallback)
        from src.llm.github_models import GitHubModelsProvider
        return GitHubModelsProvider()
    if chosen == "openai":
        from src.llm.openai_provider import OpenAIProvider
        return OpenAIProvider()
    if chosen == "anthropic":
        from src.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider()
    raise ValueError(
        f"unknown LLM_PROVIDER: {chosen!r}. "
        "Set one of: 'github' (needs GITHUB_TOKEN), 'openai' (needs OPENAI_API_KEY), "
        "'anthropic' (needs ANTHROPIC_API_KEY)."
    )
