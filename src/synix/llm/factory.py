"""LLM client factory for synix-bench.

Ported from lens-benchmark/src/lens/agent/client_factory.py with imports
adapted to the synix namespace.
"""

from __future__ import annotations

from synix.core.config import LLMConfig
from synix.core.errors import ConfigError
from synix.llm.client import BaseLLMClient, MockLLMClient


def create_llm_client(config: LLMConfig) -> BaseLLMClient:
    """Create an LLM client from configuration.

    Args:
        config: LLM configuration with provider, model, api_key, etc.

    Returns:
        A BaseLLMClient instance.

    Raises:
        ConfigError: If the provider is unknown or required config is missing.
    """
    config = config.resolve_env()

    if config.provider == "mock":
        return MockLLMClient()

    if config.provider == "openai":
        if not config.api_key:
            raise ConfigError(
                "OpenAI provider requires an API key. "
                "Set SYNIX_LLM_API_KEY or pass --api-key."
            )
        from synix.llm.openai_client import OpenAIClient

        return OpenAIClient(
            api_key=config.api_key,
            model=config.model,
            base_url=config.api_base,
            temperature=config.temperature,
            seed=config.seed,
            max_tokens=config.max_tokens,
        )

    raise ConfigError(
        f"Unknown LLM provider: {config.provider!r}. "
        f"Available providers: mock, openai"
    )
