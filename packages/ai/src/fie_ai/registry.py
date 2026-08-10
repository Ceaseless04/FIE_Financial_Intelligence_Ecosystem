"""Provider registry and routing with fallback.

Routing exists for two distinct reasons, and they are handled differently:

* **Failure** — a provider is down, timing out, or rate-limited. Retry inside
  the provider first; once its circuit trips, move on to a fallback.
* **Refusal** — the model declined on safety grounds. Retrying the identical
  request against the same model reproduces the identical refusal, so a
  refusal must skip retry entirely and go straight to a different provider.

Conflating the two produces either wasted retries or an unnecessary downgrade.
"""

from __future__ import annotations

from collections.abc import Sequence

from fie_ai.base import AIProvider
from fie_ai.contracts import CompletionRequest, CompletionResponse
from fie_common.errors import AIProviderError, ConfigurationError, FIEError
from fie_observability.logging import get_logger
from fie_schemas.health import ComponentHealth

logger = get_logger(__name__)


class ProviderRegistry:
    """Name -> provider lookup with a designated default."""

    def __init__(self) -> None:
        self._providers: dict[str, AIProvider] = {}
        self._default: str | None = None

    def register(self, provider: AIProvider, *, default: bool = False) -> None:
        self._providers[provider.name] = provider
        if default or self._default is None:
            self._default = provider.name

    def unregister(self, name: str) -> None:
        self._providers.pop(name, None)
        if self._default == name:
            self._default = next(iter(self._providers), None)

    def get(self, name: str) -> AIProvider:
        try:
            return self._providers[name]
        except KeyError as error:
            raise ConfigurationError(
                f"AI provider '{name}' is not registered",
                details={"available": sorted(self._providers)},
            ) from error

    @property
    def default(self) -> AIProvider:
        if self._default is None:
            raise ConfigurationError("no AI providers are registered")
        return self._providers[self._default]

    @property
    def names(self) -> list[str]:
        return sorted(self._providers)

    def __contains__(self, name: object) -> bool:
        return name in self._providers

    def __len__(self) -> int:
        return len(self._providers)

    async def health_check(self) -> list[ComponentHealth]:
        return [await provider.health_check() for provider in self._providers.values()]

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()


class AIRouter:
    """Routes completions to a primary provider with ordered fallbacks."""

    def __init__(
        self,
        registry: ProviderRegistry,
        *,
        primary: str | None = None,
        fallbacks: Sequence[str] = (),
    ) -> None:
        self._registry = registry
        self._primary = primary
        self._fallbacks = list(fallbacks)

    @property
    def primary(self) -> AIProvider:
        return self._registry.get(self._primary) if self._primary else self._registry.default

    @property
    def registry(self) -> ProviderRegistry:
        """The underlying registry, for health checks and shutdown.

        Routing is the router's job; owning provider lifecycle is the
        registry's. Services that need to close or probe every provider reach
        it through here rather than holding a second reference.
        """
        return self._registry

    def _chain(self, provider: str | None) -> list[AIProvider]:
        """Ordered provider chain for one request, de-duplicated."""
        names: list[str] = []
        first = provider or self._primary
        if first:
            names.append(first)
        elif self._registry.names:
            names.append(self._registry.default.name)
        for name in self._fallbacks:
            if name not in names:
                names.append(name)
        return [self._registry.get(name) for name in names]

    async def complete(
        self,
        request: CompletionRequest,
        *,
        provider: str | None = None,
        allow_fallback: bool = True,
    ) -> CompletionResponse:
        """Complete ``request``, falling back on failure or refusal.

        Raises:
            AIProviderError: if every provider in the chain fails.
        """
        chain = self._chain(provider)
        if not chain:
            raise ConfigurationError("no AI providers are registered")
        if not allow_fallback:
            chain = chain[:1]

        last_error: FIEError | None = None
        last_refusal: CompletionResponse | None = None

        for index, candidate in enumerate(chain):
            try:
                response = await candidate.complete(request)
            except FIEError as error:
                last_error = error
                logger.warning(
                    "ai_provider_failed",
                    provider=candidate.name,
                    error_code=error.code,
                    attempt=index + 1,
                    remaining=len(chain) - index - 1,
                )
                continue

            if response.is_refusal:
                last_refusal = response
                logger.warning(
                    "ai_provider_refused",
                    provider=candidate.name,
                    refusal_category=response.refusal_category,
                    remaining=len(chain) - index - 1,
                )
                continue

            return (
                response
                if index == 0
                else response.model_copy(update={"served_by": candidate.name})
            )

        # Every provider declined: surface the refusal rather than a transport
        # error, so the caller can distinguish "cannot" from "will not".
        if last_refusal is not None:
            return last_refusal
        raise AIProviderError(
            "all AI providers failed",
            details={
                "providers": [candidate.name for candidate in chain],
                "last_error": last_error.code if last_error else None,
            },
            cause=last_error,
        )

    async def count_tokens(self, request: CompletionRequest, *, provider: str | None = None) -> int:
        chain = self._chain(provider)
        if not chain:
            raise ConfigurationError("no AI providers are registered")
        return await chain[0].count_tokens(request)


__all__ = ["AIRouter", "ProviderRegistry"]
