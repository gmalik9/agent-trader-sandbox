"""Pick an LLM provider based on settings."""

from __future__ import annotations

from src.config import get_settings
from src.llm.provider import LLMProvider


def get_provider(name: str | None = None) -> LLMProvider:
    s = get_settings()
    chosen = (name or s.llm_provider or "github").lower()
    if chosen == "github":
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
