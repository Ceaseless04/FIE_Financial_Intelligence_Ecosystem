"""AI provider configuration.

Credentials and endpoints resolve from environment variables only. Each
settings class exposes ``validate_for(environment)`` so a service can assert
its production guardrails at startup rather than discovering a placeholder API
key on the first request.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, SecretStr
from pydantic_settings import SettingsConfigDict

from fie_ai.contracts import Effort
from fie_common.config import Environment, FIEBaseSettings, require_production_secret


class ProviderName(StrEnum):
    CLAUDE = "claude"
    OLLAMA = "ollama"
    VLLM = "vllm"


#: Default production model. Claude is the primary reasoning/orchestration
#: model for the ecosystem; it is never the source of truth for a calculation.
DEFAULT_CLAUDE_MODEL = "claude-opus-5"


class ClaudeSettings(FIEBaseSettings):
    """Anthropic Claude configuration (``FIE_CLAUDE_*``)."""

    model_config = SettingsConfigDict(
        env_prefix="FIE_CLAUDE_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    api_key: SecretStr | None = None
    model: str = DEFAULT_CLAUDE_MODEL
    max_tokens: int = Field(default=4096, ge=1, le=128_000)
    effort: Effort | None = None
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    #: Anthropic base URL override (proxy, gateway, or a compatible endpoint).
    base_url: str | None = None

    def validate_for(self, environment: Environment) -> None:
        require_production_secret(
            self.api_key,
            name="FIE_CLAUDE_API_KEY",
            environment=environment,
            min_length=20,
        )


class OllamaSettings(FIEBaseSettings):
    """Local Ollama configuration (``FIE_OLLAMA_*``)."""

    model_config = SettingsConfigDict(
        env_prefix="FIE_OLLAMA_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    max_tokens: int = Field(default=4096, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0)


class VLLMSettings(FIEBaseSettings):
    """vLLM configuration (``FIE_VLLM_*``). Speaks the OpenAI-compatible API."""

    model_config = SettingsConfigDict(
        env_prefix="FIE_VLLM_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    base_url: str = "http://localhost:8000/v1"
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    api_key: SecretStr | None = None
    max_tokens: int = Field(default=4096, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0)


class AISettings(FIEBaseSettings):
    """Provider selection and routing (``FIE_AI_*``)."""

    model_config = SettingsConfigDict(
        env_prefix="FIE_AI_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    default_provider: ProviderName = ProviderName.CLAUDE
    #: Tried in order when the primary provider fails or refuses.
    fallback_providers: list[ProviderName] = Field(default_factory=list)
    enabled_providers: list[ProviderName] = Field(default_factory=lambda: [ProviderName.CLAUDE])

    def validate_for(self, environment: Environment) -> None:
        if self.default_provider not in self.enabled_providers:
            raise ValueError(
                f"default_provider {self.default_provider} is not in enabled_providers"
            )
        for fallback in self.fallback_providers:
            if fallback not in self.enabled_providers:
                raise ValueError(f"fallback provider {fallback} is not in enabled_providers")
        if environment.is_production and ProviderName.CLAUDE not in self.enabled_providers:
            raise ValueError("Claude must be enabled in production")


__all__ = [
    "DEFAULT_CLAUDE_MODEL",
    "AISettings",
    "ClaudeSettings",
    "OllamaSettings",
    "ProviderName",
    "VLLMSettings",
]
