"""OpenAI BYO-key provider (same OpenAI-compatible request shape as GitHub Models)."""

from __future__ import annotations

from src.config import get_settings
from src.llm.github_models import GitHubModelsProvider


class OpenAIProvider(GitHubModelsProvider):
    """OpenAI uses the same chat-completions schema; reuse the parser/encoder."""

    name = "openai"

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 *, endpoint: str = "https://api.openai.com/v1/chat/completions",
                 timeout: float = 60.0) -> None:
        s = get_settings()
        key = api_key or s.openai_api_key
        if not key:
            raise ValueError("OPENAI_API_KEY not set")
        # Cleaner default model for OpenAI:
        chosen = model or (s.llm_model if not s.llm_model.startswith("openai/") else
                            s.llm_model.split("/", 1)[1])
        super().__init__(token=key, model=chosen or "gpt-4o-mini",
                         endpoint=endpoint, timeout=timeout)
        self.name = "openai"
