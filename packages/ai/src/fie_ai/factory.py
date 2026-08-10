"""Build a configured provider registry and router from settings."""

from __future__ import annotations

from fie_ai.config import (
    AISettings,
    ClaudeSettings,
    OllamaSettings,
    ProviderName,
    VLLMSettings,
)
from fie_ai.registry import AIRouter, ProviderRegistry
from fie_common.config import Environment
from fie_common.errors import ConfigurationError
from fie_observability.logging import get_logger

logger = get_logger(__name__)


def build_registry(
    settings: AISettings | None = None,
    *,
    environment: Environment = Environment.DEVELOPMENT,
    claude_settings: ClaudeSettings | None = None,
    ollama_settings: OllamaSettings | None = None,
    vllm_settings: VLLMSettings | None = None,
) -> ProviderRegistry:
    """Instantiate every enabled provider.

    Providers are imported lazily so a service that only uses Ollama in local
    development does not pay the cost of importing the Anthropic SDK, and a
    misconfigured optional provider cannot prevent startup.
    """
    settings = settings or AISettings()
    settings.validate_for(environment)

    registry = ProviderRegistry()

    for name in settings.enabled_providers:
        is_default = name == settings.default_provider
        try:
            if name is ProviderName.CLAUDE:
                from fie_ai.providers.claude import ClaudeProvider

                claude_config = claude_settings or ClaudeSettings()
                claude_config.validate_for(environment)
                registry.register(ClaudeProvider(claude_config), default=is_default)
            elif name is ProviderName.OLLAMA:
                from fie_ai.providers.ollama import OllamaProvider

                registry.register(
                    OllamaProvider(ollama_settings or OllamaSettings()), default=is_default
                )
            elif name is ProviderName.VLLM:
                from fie_ai.providers.vllm import VLLMProvider

                registry.register(VLLMProvider(vllm_settings or VLLMSettings()), default=is_default)
        except ConfigurationError:
            # The default provider is load-bearing; an optional one is not.
            if is_default:
                raise
            logger.warning("ai_provider_skipped", provider=str(name))

    if len(registry) == 0:
        raise ConfigurationError(
            "no AI providers could be constructed",
            details={"enabled": [str(name) for name in settings.enabled_providers]},
        )
    return registry


def build_router(
    settings: AISettings | None = None,
    *,
    environment: Environment = Environment.DEVELOPMENT,
    registry: ProviderRegistry | None = None,
) -> AIRouter:
    """Build the router agents and services call."""
    settings = settings or AISettings()
    registry = registry or build_registry(settings, environment=environment)
    return AIRouter(
        registry,
        primary=str(settings.default_provider),
        fallbacks=[str(name) for name in settings.fallback_providers],
    )


__all__ = ["build_registry", "build_router"]
