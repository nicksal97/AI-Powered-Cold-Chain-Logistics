"""One chat-model factory for every provider.

OpenAI, DeepSeek and Ollama all expose the OpenAI chat-completions API, so a single
client with a different base_url covers all three and keeps tool-calling behaviour
identical across them.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from coldchain.config import Settings

DEFAULT_BASE_URLS = {
    "openai": None,
    "deepseek": "https://api.deepseek.com",
    "ollama": "http://localhost:11434/v1",
}


def build_llm(settings: Settings) -> BaseChatModel:
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    if settings.llm_provider == "ollama":
        api_key = api_key or "ollama"  # Ollama ignores it, the client requires one
    if not api_key:
        raise ValueError("COLDCHAIN_LLM_API_KEY is required for provider " + settings.llm_provider)
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=api_key,
        base_url=settings.llm_base_url or DEFAULT_BASE_URLS[settings.llm_provider],
        timeout=settings.llm_timeout_s,
        max_retries=2,
    )
