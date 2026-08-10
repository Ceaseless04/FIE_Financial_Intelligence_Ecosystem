"""Unit tests for the provider registry, router fallback, and factory."""

from __future__ import annotations

import pytest

from fie_ai.config import AISettings, ClaudeSettings, OllamaSettings, ProviderName
from fie_ai.contracts import CompletionRequest, Message, Role, StopReason
from fie_ai.factory import build_registry, build_router
from fie_ai.registry import AIRouter, ProviderRegistry
from fie_common.config import Environment, SecretStr
from fie_common.errors import AIProviderError, ConfigurationError, DatabaseError
from fie_testing.fakes import FakeAIProvider

pytestmark = pytest.mark.unit


def request(prompt: str = "Analyze this.") -> CompletionRequest:
    return CompletionRequest(messages=[Message(role=Role.USER, content=prompt)])


def fake(name: str, *responses: object) -> FakeAIProvider:
    provider = FakeAIProvider(provider_name=name, model=f"{name}-model")
    if responses:
        provider.enqueue(*responses)  # type: ignore[arg-type]
    return provider


class TestProviderRegistry:
    def test_first_registration_becomes_the_default(self) -> None:
        registry = ProviderRegistry()
        primary = fake("claude")
        registry.register(primary)

        assert registry.default is primary

    def test_explicit_default_wins(self) -> None:
        registry = ProviderRegistry()
        registry.register(fake("ollama"))
        claude = fake("claude")
        registry.register(claude, default=True)

        assert registry.default is claude

    def test_lookup_by_name(self) -> None:
        registry = ProviderRegistry()
        ollama = fake("ollama")
        registry.register(ollama)

        assert registry.get("ollama") is ollama

    def test_unknown_provider_raises_with_the_available_list(self) -> None:
        registry = ProviderRegistry()
        registry.register(fake("claude"))

        with pytest.raises(ConfigurationError) as exc_info:
            registry.get("missing")

        assert exc_info.value.details["available"] == ["claude"]

    def test_default_on_an_empty_registry_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="no AI providers"):
            _ = ProviderRegistry().default

    def test_unregister_reassigns_the_default(self) -> None:
        registry = ProviderRegistry()
        registry.register(fake("claude"), default=True)
        registry.register(fake("ollama"))
        registry.unregister("claude")

        assert registry.default.name == "ollama"

    def test_membership_and_length(self) -> None:
        registry = ProviderRegistry()
        registry.register(fake("claude"))

        assert "claude" in registry
        assert len(registry) == 1
        assert registry.names == ["claude"]

    async def test_health_check_covers_every_provider(self) -> None:
        registry = ProviderRegistry()
        registry.register(fake("claude"))
        registry.register(fake("ollama"))

        health = await registry.health_check()

        assert {component.name for component in health} == {"ai:claude", "ai:ollama"}

    async def test_aclose_closes_every_provider(self) -> None:
        registry = ProviderRegistry()
        claude, ollama = fake("claude"), fake("ollama")
        registry.register(claude)
        registry.register(ollama)

        await registry.aclose()

        assert claude.closed and ollama.closed


class TestRouterHappyPath:
    async def test_routes_to_the_primary_provider(self) -> None:
        registry = ProviderRegistry()
        claude = fake("claude")
        claude.enqueue_text("primary answer")
        registry.register(claude)
        router = AIRouter(registry, primary="claude")

        response = await router.complete(request())

        assert response.text == "primary answer"
        assert response.served_by is None, "no fallback was needed"

    async def test_explicit_provider_selection_overrides_the_primary(self) -> None:
        registry = ProviderRegistry()
        registry.register(fake("claude"))
        ollama = fake("ollama")
        ollama.enqueue_text("from ollama")
        registry.register(ollama)
        router = AIRouter(registry, primary="claude")

        response = await router.complete(request(), provider="ollama")

        assert response.text == "from ollama"

    async def test_primary_property_resolves_the_provider(self) -> None:
        registry = ProviderRegistry()
        registry.register(fake("claude"))
        router = AIRouter(registry, primary="claude")

        assert router.primary.name == "claude"

    async def test_count_tokens_uses_the_head_of_the_chain(self) -> None:
        registry = ProviderRegistry()
        registry.register(fake("claude"))
        router = AIRouter(registry, primary="claude")

        assert await router.count_tokens(request()) > 0


class TestRouterFallback:
    async def test_falls_back_when_the_primary_fails(self) -> None:
        registry = ProviderRegistry()
        claude = fake("claude", DatabaseError("provider down"))
        registry.register(claude)
        ollama = fake("ollama")
        ollama.enqueue_text("fallback answer")
        registry.register(ollama)
        router = AIRouter(registry, primary="claude", fallbacks=["ollama"])

        response = await router.complete(request())

        assert response.text == "fallback answer"
        assert response.served_by == "ollama", "the response records which provider served it"

    async def test_falls_back_on_a_refusal(self) -> None:
        # A refusal is deterministic on the same model, so the only useful
        # recovery is a different provider.
        registry = ProviderRegistry()
        claude = fake("claude")
        claude.enqueue(
            claude.make_response("", stop_reason=StopReason.REFUSAL, refusal_category="cyber")
        )
        registry.register(claude)
        ollama = fake("ollama")
        ollama.enqueue_text("answered by the fallback")
        registry.register(ollama)
        router = AIRouter(registry, primary="claude", fallbacks=["ollama"])

        response = await router.complete(request())

        assert response.text == "answered by the fallback"
        assert response.is_refusal is False

    async def test_returns_the_refusal_when_every_provider_declines(self) -> None:
        # Surfacing the refusal rather than a transport error lets the caller
        # distinguish "cannot" from "will not".
        registry = ProviderRegistry()
        for name in ("claude", "ollama"):
            provider = fake(name)
            provider.enqueue(
                provider.make_response("", stop_reason=StopReason.REFUSAL, refusal_category="cyber")
            )
            registry.register(provider)
        router = AIRouter(registry, primary="claude", fallbacks=["ollama"])

        response = await router.complete(request())

        assert response.is_refusal is True
        assert response.refusal_category == "cyber"

    async def test_raises_when_every_provider_fails(self) -> None:
        registry = ProviderRegistry()
        registry.register(fake("claude", DatabaseError("down")))
        registry.register(fake("ollama", DatabaseError("also down")))
        router = AIRouter(registry, primary="claude", fallbacks=["ollama"])

        with pytest.raises(AIProviderError, match="all AI providers failed") as exc_info:
            await router.complete(request())

        assert exc_info.value.details["providers"] == ["claude", "ollama"]

    async def test_fallback_can_be_disabled_per_call(self) -> None:
        registry = ProviderRegistry()
        registry.register(fake("claude", DatabaseError("down")))
        ollama = fake("ollama")
        ollama.enqueue_text("never reached")
        registry.register(ollama)
        router = AIRouter(registry, primary="claude", fallbacks=["ollama"])

        with pytest.raises(AIProviderError):
            await router.complete(request(), allow_fallback=False)

        assert ollama.call_count == 0

    async def test_the_chain_is_deduplicated(self) -> None:
        registry = ProviderRegistry()
        claude = fake("claude", DatabaseError("down"))
        registry.register(claude)
        router = AIRouter(registry, primary="claude", fallbacks=["claude"])

        with pytest.raises(AIProviderError) as exc_info:
            await router.complete(request())

        assert exc_info.value.details["providers"] == ["claude"]

    async def test_falls_through_multiple_fallbacks_in_order(self) -> None:
        registry = ProviderRegistry()
        registry.register(fake("claude", DatabaseError("down")))
        registry.register(fake("ollama", DatabaseError("down")))
        vllm = fake("vllm")
        vllm.enqueue_text("third time lucky")
        registry.register(vllm)
        router = AIRouter(registry, primary="claude", fallbacks=["ollama", "vllm"])

        response = await router.complete(request())

        assert response.text == "third time lucky"
        assert response.served_by == "vllm"

    async def test_empty_registry_raises_configuration_error(self) -> None:
        router = AIRouter(ProviderRegistry())

        with pytest.raises(ConfigurationError):
            await router.complete(request())

    async def test_router_without_an_explicit_primary_uses_the_registry_default(self) -> None:
        registry = ProviderRegistry()
        claude = fake("claude")
        claude.enqueue_text("default provider")
        registry.register(claude)
        router = AIRouter(registry)

        assert (await router.complete(request())).text == "default provider"


class TestAISettings:
    def test_default_provider_must_be_enabled(self) -> None:
        settings = AISettings(
            default_provider=ProviderName.OLLAMA, enabled_providers=[ProviderName.CLAUDE]
        )

        with pytest.raises(ValueError, match="not in enabled_providers"):
            settings.validate_for(Environment.DEVELOPMENT)

    def test_fallbacks_must_be_enabled(self) -> None:
        settings = AISettings(
            default_provider=ProviderName.CLAUDE,
            enabled_providers=[ProviderName.CLAUDE],
            fallback_providers=[ProviderName.VLLM],
        )

        with pytest.raises(ValueError, match="fallback provider"):
            settings.validate_for(Environment.DEVELOPMENT)

    def test_claude_must_be_enabled_in_production(self) -> None:
        settings = AISettings(
            default_provider=ProviderName.OLLAMA, enabled_providers=[ProviderName.OLLAMA]
        )

        with pytest.raises(ValueError, match="Claude must be enabled in production"):
            settings.validate_for(Environment.PRODUCTION)

    def test_a_valid_configuration_passes(self) -> None:
        AISettings(
            default_provider=ProviderName.CLAUDE,
            enabled_providers=[ProviderName.CLAUDE, ProviderName.OLLAMA],
            fallback_providers=[ProviderName.OLLAMA],
        ).validate_for(Environment.PRODUCTION)


class TestFactory:
    def test_builds_the_enabled_providers(self) -> None:
        registry = build_registry(
            AISettings(
                default_provider=ProviderName.OLLAMA,
                enabled_providers=[ProviderName.OLLAMA, ProviderName.VLLM],
            ),
            environment=Environment.DEVELOPMENT,
        )

        assert set(registry.names) == {"ollama", "vllm"}
        assert registry.default.name == "ollama"

    def test_a_missing_claude_key_fails_when_claude_is_the_default(self) -> None:
        with pytest.raises(ConfigurationError):
            build_registry(
                AISettings(
                    default_provider=ProviderName.CLAUDE,
                    enabled_providers=[ProviderName.CLAUDE],
                ),
                environment=Environment.DEVELOPMENT,
                claude_settings=ClaudeSettings(api_key=None),
            )

    def test_an_optional_provider_that_cannot_start_is_skipped(self) -> None:
        # A misconfigured optional provider must not prevent startup.
        registry = build_registry(
            AISettings(
                default_provider=ProviderName.OLLAMA,
                enabled_providers=[ProviderName.OLLAMA, ProviderName.CLAUDE],
            ),
            environment=Environment.DEVELOPMENT,
            claude_settings=ClaudeSettings(api_key=None),
        )

        assert registry.names == ["ollama"]

    def test_claude_is_built_when_a_key_is_present(self) -> None:
        registry = build_registry(
            AISettings(
                default_provider=ProviderName.CLAUDE,
                enabled_providers=[ProviderName.CLAUDE],
            ),
            environment=Environment.DEVELOPMENT,
            claude_settings=ClaudeSettings(api_key=SecretStr("sk-ant-test-key-value")),
        )

        assert registry.default.name == "claude"

    def test_build_router_wires_primary_and_fallbacks(self) -> None:
        settings = AISettings(
            default_provider=ProviderName.OLLAMA,
            enabled_providers=[ProviderName.OLLAMA, ProviderName.VLLM],
            fallback_providers=[ProviderName.VLLM],
        )
        router = build_router(settings, environment=Environment.DEVELOPMENT)

        assert router.primary.name == "ollama"

    def test_build_router_accepts_a_prebuilt_registry(self) -> None:
        registry = ProviderRegistry()
        registry.register(fake("ollama"))
        router = build_router(
            AISettings(
                default_provider=ProviderName.OLLAMA,
                enabled_providers=[ProviderName.OLLAMA],
            ),
            registry=registry,
        )

        assert router.primary.name == "ollama"

    def test_custom_ollama_settings_are_honored(self) -> None:
        registry = build_registry(
            AISettings(
                default_provider=ProviderName.OLLAMA,
                enabled_providers=[ProviderName.OLLAMA],
            ),
            ollama_settings=OllamaSettings(model="mistral:7b"),
        )

        assert registry.get("ollama").default_model == "mistral:7b"
