"""
LLM Router
-----------
Provider-agnostic LLM abstraction with:
  - Priority-ordered fallback chain (primary → secondary → local)
  - Retry with exponential backoff
  - Token counting and budget enforcement
  - Response caching (optional, Redis-backed in production)

Design: Adding a new LLM provider = implement BaseLLMProvider + register.
        The orchestrator never imports any provider directly.

Fallback chain rationale (enterprise context):
  Azure OpenAI → primary (data residency, SLA)
  OpenAI       → fallback (if Azure quota exhausted)
  Local model  → emergency fallback (air-gapped / outage scenario)
"""

import time
import logging
import hashlib
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE = 2  # seconds


# ── Base Provider Interface ────────────────────────────────────────────────────

class BaseLLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, max_tokens: int) -> tuple[str, int]:
        """Returns (answer_text, tokens_used)."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass


# ── Concrete Providers ─────────────────────────────────────────────────────────

class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI — preferred for GCC/enterprise (data residency compliance)."""

    name = "azure_openai"

    def __init__(self, config: dict):
        from langchain_openai import AzureChatOpenAI
        self._llm = AzureChatOpenAI(
            azure_deployment=config["deployment"],
            azure_endpoint=config["endpoint"],
            api_version=config.get("api_version", "2024-02-01"),
            temperature=config.get("temperature", 0),
            max_tokens=config.get("max_tokens", 1024),
        )

    def complete(self, prompt: str, max_tokens: int = 1024) -> tuple[str, int]:
        from langchain_core.messages import HumanMessage
        response = self._llm.invoke([HumanMessage(content=prompt)])
        tokens = response.usage_metadata.get("total_tokens", 0) if hasattr(response, "usage_metadata") else 0
        return response.content, tokens


class OpenAIProvider(BaseLLMProvider):
    """OpenAI direct — secondary fallback."""

    name = "openai"

    def __init__(self, config: dict):
        from langchain_openai import ChatOpenAI
        self._llm = ChatOpenAI(
            model=config.get("model", "gpt-4o"),
            temperature=config.get("temperature", 0),
            max_tokens=config.get("max_tokens", 1024),
        )

    def complete(self, prompt: str, max_tokens: int = 1024) -> tuple[str, int]:
        from langchain_core.messages import HumanMessage
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return response.content, 0  # token counting varies by version


class LocalProvider(BaseLLMProvider):
    """
    Local HuggingFace model — emergency fallback / air-gapped environments.
    Lower quality but zero external dependency.
    """

    name = "local"

    def __init__(self, config: dict):
        self._model_name = config.get("model", "microsoft/phi-2")
        self._pipeline = None  # lazy load

    def _ensure_loaded(self):
        if self._pipeline is None:
            from transformers import pipeline
            logger.info("Loading local model: %s", self._model_name)
            self._pipeline = pipeline("text-generation", model=self._model_name, max_new_tokens=512)

    def complete(self, prompt: str, max_tokens: int = 512) -> tuple[str, int]:
        self._ensure_loaded()
        result = self._pipeline(prompt, max_new_tokens=max_tokens)
        answer = result[0]["generated_text"][len(prompt):]
        return answer.strip(), 0


# ── Router ─────────────────────────────────────────────────────────────────────

PROVIDER_REGISTRY = {
    "azure_openai": AzureOpenAIProvider,
    "openai": OpenAIProvider,
    "local": LocalProvider,
}


class LLMRouter:
    """
    Tries providers in priority order. On failure, logs and moves to next.
    All retries use exponential backoff to handle transient rate limits.

    Cache: optional in-memory dict (replace with Redis in production).
    """

    def __init__(self, config: dict):
        self._chain: list[BaseLLMProvider] = []
        self._cache: dict[str, tuple[str, int]] = {}
        self._cache_enabled = config.get("cache", False)
        self._max_tokens = config.get("max_tokens", 1024)

        for provider_cfg in config.get("providers", []):
            name = provider_cfg["name"]
            if name not in PROVIDER_REGISTRY:
                raise ValueError(f"Unknown LLM provider: {name}")
            try:
                provider = PROVIDER_REGISTRY[name](provider_cfg)
                self._chain.append(provider)
                logger.info("Registered LLM provider: %s", name)
            except Exception as exc:
                logger.warning("Could not initialise provider %s: %s", name, exc)

        if not self._chain:
            raise RuntimeError("No LLM providers could be initialised. Check config.")

    def complete(self, prompt: str) -> tuple[str, int]:
        # Cache lookup
        if self._cache_enabled:
            cache_key = hashlib.sha256(prompt.encode()).hexdigest()
            if cache_key in self._cache:
                logger.debug("Cache hit for prompt hash %s", cache_key[:8])
                return self._cache[cache_key]

        last_error = None
        for provider in self._chain:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    logger.info("LLM call → %s (attempt %d)", provider.name, attempt)
                    result = provider.complete(prompt, self._max_tokens)
                    if self._cache_enabled:
                        self._cache[cache_key] = result
                    return result
                except Exception as exc:
                    last_error = exc
                    wait = BACKOFF_BASE ** attempt
                    logger.warning(
                        "Provider %s failed (attempt %d/%d): %s. Retrying in %ds.",
                        provider.name, attempt, MAX_RETRIES, exc, wait
                    )
                    time.sleep(wait)
            logger.error("Provider %s exhausted retries, trying next.", provider.name)

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
