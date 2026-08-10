"""Unit tests for the Claude provider.

These exercise the real translation logic against a client double that
reproduces the Anthropic SDK's response shapes, so the behaviours that matter
are genuinely tested rather than mocked away: sampling parameters being
dropped, adaptive thinking, cache breakpoints, refusal handling, and the
mapping of SDK exceptions onto the platform error hierarchy.
"""

from __future__ import annotations

import anthropic
import httpx
import pytest

from fie_ai.config import ClaudeSettings
from fie_ai.contracts import (
    CompletionRequest,
    Effort,
    Message,
    Role,
    StopReason,
    ThinkingMode,
)
from fie_ai.providers.claude import ClaudeProvider
from fie_common.config import Environment, SecretStr
from fie_common.errors import (
    AIProviderError,
    AIProviderRateLimitError,
    AIProviderTimeoutError,
    AuthenticationError,
    ConfigurationError,
    ValidationError,
)
from fie_common.resilience import ResiliencePolicy, RetryPolicy
from fie_testing.fakes import (
    FakeAnthropicClient,
    FakeMessage,
    FakeTextBlock,
    FakeUsage,
    make_refusal_message,
    make_text_message,
)

pytestmark = pytest.mark.unit


def no_retry_policy() -> ResiliencePolicy:
    """Single-attempt policy.

    Error-translation tests assert on the exception a single call produces;
    with the default policy a retryable error would be retried and the fake's
    next queued response returned instead. Retry behaviour is covered
    separately in ``TestRetryIntegration``.
    """
    return ResiliencePolicy(timeout_seconds=5.0, retry=RetryPolicy(max_attempts=1), breaker=None)


def make_provider(
    client: FakeAnthropicClient | None = None,
    *,
    resilience: ResiliencePolicy | None = None,
    **settings_overrides: object,
) -> tuple[ClaudeProvider, FakeAnthropicClient]:
    fake = client or FakeAnthropicClient()
    settings = ClaudeSettings(api_key=SecretStr("sk-ant-test-key"), **settings_overrides)  # type: ignore[arg-type]
    return ClaudeProvider(settings, client=fake, resilience=resilience or no_retry_policy()), fake


def user_request(prompt: str = "Summarize this filing.", **kwargs: object) -> CompletionRequest:
    return CompletionRequest(
        messages=[Message(role=Role.USER, content=prompt)],
        **kwargs,  # type: ignore[arg-type]
    )


class TestConstruction:
    def test_requires_an_api_key_when_no_client_is_injected(self) -> None:
        with pytest.raises(ConfigurationError, match="FIE_CLAUDE_API_KEY"):
            ClaudeProvider(ClaudeSettings(api_key=None))

    def test_an_injected_client_bypasses_the_api_key_requirement(self) -> None:
        provider = ClaudeProvider(ClaudeSettings(api_key=None), client=FakeAnthropicClient())

        assert provider.name == "claude"

    def test_default_model_comes_from_settings(self) -> None:
        provider, _ = make_provider(model="claude-opus-5")

        assert provider.default_model == "claude-opus-5"

    def test_capabilities_report_no_sampling_support(self) -> None:
        provider, _ = make_provider()

        assert provider.capabilities.supports_sampling_params is False
        assert provider.capabilities.supports_thinking is True
        assert provider.capabilities.supports_prompt_caching is True

    def test_production_validation_rejects_a_missing_key(self) -> None:
        provider = ClaudeProvider(ClaudeSettings(api_key=None), client=FakeAnthropicClient())

        with pytest.raises(ConfigurationError):
            provider.validate_for(Environment.PRODUCTION)


class TestRequestConstruction:
    async def test_sends_adaptive_thinking_by_default(self) -> None:
        provider, fake = make_provider()
        await provider.complete(user_request())

        assert fake.last_request["thinking"] == {"type": "adaptive"}

    async def test_never_sends_budget_tokens(self) -> None:
        # budget_tokens is removed on current models and returns a 400.
        provider, fake = make_provider()
        await provider.complete(user_request())

        assert "budget_tokens" not in str(fake.last_request)

    async def test_drops_temperature_for_models_that_reject_it(self) -> None:
        # A caller may set temperature on the neutral contract; forwarding it
        # to Claude Opus 5 would be a 400, so the provider absorbs it.
        provider, fake = make_provider(model="claude-opus-5")
        await provider.complete(user_request(temperature=0.7))

        assert "temperature" not in fake.last_request

    async def test_forwards_temperature_for_legacy_models(self) -> None:
        provider, fake = make_provider(model="claude-3-haiku-20240307")
        await provider.complete(user_request(temperature=0.7))

        assert fake.last_request["temperature"] == 0.7

    async def test_system_prompt_carries_a_cache_breakpoint(self) -> None:
        provider, fake = make_provider()
        await provider.complete(user_request(system="You are a filing analyst."))

        system = fake.last_request["system"]
        assert system[0]["cache_control"] == {"type": "ephemeral"}
        assert system[0]["text"] == "You are a filing analyst."

    async def test_caching_can_be_disabled(self) -> None:
        provider, fake = make_provider()
        await provider.complete(
            user_request(system="You are an analyst.", cache_system_prompt=False)
        )

        assert fake.last_request["system"] == "You are an analyst."

    async def test_no_system_key_when_no_system_prompt(self) -> None:
        provider, fake = make_provider()
        await provider.complete(user_request())

        assert "system" not in fake.last_request

    async def test_effort_goes_inside_output_config(self) -> None:
        provider, fake = make_provider()
        await provider.complete(user_request(effort=Effort.HIGH))

        assert fake.last_request["output_config"] == {"effort": "high"}

    async def test_response_schema_becomes_a_json_schema_format(self) -> None:
        provider, fake = make_provider()
        schema = {"type": "object", "properties": {"ticker": {"type": "string"}}}
        await provider.complete(user_request(response_schema=schema))

        assert fake.last_request["output_config"]["format"] == {
            "type": "json_schema",
            "schema": schema,
        }

    async def test_effort_and_schema_share_one_output_config(self) -> None:
        provider, fake = make_provider()
        await provider.complete(
            user_request(effort=Effort.MEDIUM, response_schema={"type": "object"})
        )

        output_config = fake.last_request["output_config"]
        assert output_config["effort"] == "medium"
        assert "format" in output_config

    async def test_no_output_config_when_neither_is_requested(self) -> None:
        provider, fake = make_provider()
        await provider.complete(user_request())

        assert "output_config" not in fake.last_request

    async def test_stop_sequences_are_forwarded(self) -> None:
        provider, fake = make_provider()
        await provider.complete(user_request(stop_sequences=["END"]))

        assert fake.last_request["stop_sequences"] == ["END"]

    async def test_request_model_overrides_the_default(self) -> None:
        provider, fake = make_provider(model="claude-opus-5")
        await provider.complete(user_request(model="claude-sonnet-5"))

        assert fake.last_request["model"] == "claude-sonnet-5"

    @pytest.mark.parametrize("effort", [Effort.XHIGH, Effort.MAX])
    async def test_disabled_thinking_above_high_effort_is_rejected(self, effort: Effort) -> None:
        # Rejected by the API with a 400; caught locally with a clearer message.
        provider, _ = make_provider()

        with pytest.raises(ValidationError, match="thinking cannot be disabled"):
            await provider.complete(user_request(thinking=ThinkingMode.DISABLED, effort=effort))

    @pytest.mark.parametrize("effort", [Effort.LOW, Effort.MEDIUM, Effort.HIGH, None])
    async def test_disabled_thinking_is_allowed_at_high_effort_or_below(
        self, effort: Effort | None
    ) -> None:
        provider, fake = make_provider()
        await provider.complete(user_request(thinking=ThinkingMode.DISABLED, effort=effort))

        assert fake.last_request["thinking"] == {"type": "disabled"}


class TestStreamingThreshold:
    async def test_small_requests_use_the_non_streaming_path(self) -> None:
        provider, fake = make_provider()
        await provider.complete(user_request(max_tokens=4096))

        assert len(fake.requests) == 1

    async def test_large_requests_are_reassembled_from_a_stream(self) -> None:
        provider, _ = make_provider(client=FakeAnthropicClient([make_text_message("long output")]))
        response = await provider.complete(user_request(max_tokens=64_000))

        assert response.text == "long output"

    async def test_stream_yields_text_deltas(self) -> None:
        provider, _ = make_provider(
            client=FakeAnthropicClient([make_text_message("alpha beta gamma")])
        )

        chunks = [chunk async for chunk in provider.stream(user_request())]

        assert "".join(chunks).strip() == "alpha beta gamma"


class TestResponseTranslation:
    async def test_extracts_text_and_usage(self) -> None:
        provider, _ = make_provider(
            client=FakeAnthropicClient(
                [make_text_message("Revenue grew 12%.", input_tokens=120, output_tokens=45)]
            )
        )
        response = await provider.complete(user_request())

        assert response.text == "Revenue grew 12%."
        assert response.usage.input_tokens == 120
        assert response.usage.output_tokens == 45
        assert response.provider == "claude"

    async def test_cache_token_kinds_are_preserved_separately(self) -> None:
        provider, _ = make_provider(
            client=FakeAnthropicClient(
                [make_text_message("cached", cache_read=900, cache_write=100)]
            )
        )
        response = await provider.complete(user_request())

        assert response.usage.cache_read_input_tokens == 900
        assert response.usage.cache_creation_input_tokens == 100

    async def test_thinking_blocks_are_skipped_when_extracting_text(self) -> None:
        provider, _ = make_provider(
            client=FakeAnthropicClient([make_text_message("answer", with_thinking=True)])
        )
        response = await provider.complete(user_request())

        assert response.text == "answer"

    async def test_multiple_text_blocks_are_concatenated(self) -> None:
        message = FakeMessage(
            content=[FakeTextBlock(text="part one. "), FakeTextBlock(text="part two.")],
            usage=FakeUsage(1, 2),
        )
        provider, _ = make_provider(client=FakeAnthropicClient([message]))
        response = await provider.complete(user_request())

        assert response.text == "part one. part two."

    async def test_refusal_is_surfaced_without_indexing_empty_content(self) -> None:
        # A refusal is HTTP 200 with an empty content array. Code that reads
        # content[0] unconditionally would raise here.
        provider, _ = make_provider(
            client=FakeAnthropicClient([make_refusal_message(category="cyber")])
        )
        response = await provider.complete(user_request())

        assert response.is_refusal is True
        assert response.refusal_category == "cyber"
        assert response.text == ""

    async def test_max_tokens_stop_reason_is_mapped(self) -> None:
        message = FakeMessage(
            content=[FakeTextBlock(text="truncated")],
            stop_reason="max_tokens",
            usage=FakeUsage(1, 2),
        )
        provider, _ = make_provider(client=FakeAnthropicClient([message]))
        response = await provider.complete(user_request())

        assert response.is_truncated is True

    async def test_unknown_stop_reason_degrades_to_end_turn(self) -> None:
        message = FakeMessage(
            content=[FakeTextBlock(text="x")], stop_reason="something_new", usage=FakeUsage()
        )
        provider, _ = make_provider(client=FakeAnthropicClient([message]))
        response = await provider.complete(user_request())

        assert response.stop_reason is StopReason.END_TURN

    async def test_missing_usage_yields_zeroed_tokens(self) -> None:
        message = FakeMessage(content=[FakeTextBlock(text="x")])
        message.usage = None  # type: ignore[assignment]
        provider, _ = make_provider(client=FakeAnthropicClient([message]))
        response = await provider.complete(user_request())

        assert response.usage.total_tokens == 0


class TestTokenCounting:
    async def test_delegates_to_the_count_tokens_endpoint(self) -> None:
        provider, _ = make_provider(client=FakeAnthropicClient(token_count=1234))

        assert await provider.count_tokens(user_request()) == 1234

    async def test_includes_the_system_prompt_in_the_count(self) -> None:
        provider, fake = make_provider()
        await provider.count_tokens(user_request(system="You are an analyst."))

        assert fake.token_count_requests[-1]["system"] == "You are an analyst."


class TestErrorTranslation:
    def _response(self, status: int, headers: dict[str, str] | None = None) -> httpx.Response:
        return httpx.Response(
            status, headers=headers or {}, request=httpx.Request("POST", "https://api.test")
        )

    async def test_timeout_maps_to_a_retryable_timeout_error(self) -> None:
        error = anthropic.APITimeoutError(request=httpx.Request("POST", "https://api.test"))
        provider, _ = make_provider(client=FakeAnthropicClient([error]))

        with pytest.raises(AIProviderTimeoutError) as exc_info:
            await provider.complete(user_request())

        assert exc_info.value.retryable is True

    async def test_rate_limit_maps_to_a_retryable_rate_limit_error(self) -> None:
        error = anthropic.RateLimitError(
            "rate limited", response=self._response(429, {"retry-after": "5"}), body=None
        )
        provider, _ = make_provider(client=FakeAnthropicClient([error]))

        with pytest.raises(AIProviderRateLimitError) as exc_info:
            await provider.complete(user_request())

        assert exc_info.value.retryable is True

    async def test_authentication_failure_is_not_retryable(self) -> None:
        error = anthropic.AuthenticationError("bad key", response=self._response(401), body=None)
        provider, _ = make_provider(client=FakeAnthropicClient([error]))

        with pytest.raises(AuthenticationError) as exc_info:
            await provider.complete(user_request())

        assert exc_info.value.retryable is False

    async def test_bad_request_is_not_retryable(self) -> None:
        error = anthropic.BadRequestError(
            "invalid parameter", response=self._response(400), body=None
        )
        provider, _ = make_provider(client=FakeAnthropicClient([error]))

        with pytest.raises(ValidationError) as exc_info:
            await provider.complete(user_request())

        assert exc_info.value.retryable is False

    async def test_connection_error_is_retryable(self) -> None:
        error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.test"))
        provider, _ = make_provider(client=FakeAnthropicClient([error]))

        with pytest.raises(AIProviderError) as exc_info:
            await provider.complete(user_request())

        assert exc_info.value.retryable is True

    async def test_unexpected_exceptions_are_wrapped(self) -> None:
        provider, _ = make_provider(client=FakeAnthropicClient([RuntimeError("weird")]))

        with pytest.raises(AIProviderError, match="Claude call failed"):
            await provider.complete(user_request())


class TestRetryIntegration:
    """The base class wraps every provider call in the resilience policy."""

    @staticmethod
    def _fast_retry() -> ResiliencePolicy:
        return ResiliencePolicy(
            timeout_seconds=5.0,
            retry=RetryPolicy(max_attempts=3, initial_backoff_seconds=0.0, jitter=0.0),
            breaker=None,
        )

    async def test_a_retryable_error_is_retried_and_can_succeed(self) -> None:
        timeout = anthropic.APITimeoutError(request=httpx.Request("POST", "https://api.test"))
        fake = FakeAnthropicClient([timeout, make_text_message("recovered")])
        provider, _ = make_provider(client=fake, resilience=self._fast_retry())

        response = await provider.complete(user_request())

        assert response.text == "recovered"
        assert len(fake.requests) == 2

    async def test_a_non_retryable_error_is_not_retried(self) -> None:
        error = anthropic.BadRequestError(
            "invalid parameter",
            response=httpx.Response(400, request=httpx.Request("POST", "https://api.test")),
            body=None,
        )
        fake = FakeAnthropicClient([error, make_text_message("never reached")])
        provider, _ = make_provider(client=fake, resilience=self._fast_retry())

        with pytest.raises(ValidationError):
            await provider.complete(user_request())

        assert len(fake.requests) == 1, "a 400 must not be retried"

    async def test_a_refusal_is_not_retried(self) -> None:
        # A refusal is a successful call with a declined body; retrying it
        # reproduces the same refusal. Routing must switch providers instead.
        fake = FakeAnthropicClient([make_refusal_message(), make_text_message("second")])
        provider, _ = make_provider(client=fake, resilience=self._fast_retry())

        response = await provider.complete(user_request())

        assert response.is_refusal is True
        assert len(fake.requests) == 1


class TestLifecycle:
    async def test_ping_uses_the_free_token_counting_endpoint(self) -> None:
        provider, fake = make_provider()
        await provider._ping()

        assert len(fake.token_count_requests) == 1

    async def test_health_check_reports_healthy(self) -> None:
        provider, _ = make_provider()
        health = await provider.health_check()

        assert health.name == "ai:claude"
        assert health.status.is_serving is True
        assert health.required is False

    async def test_health_check_reports_unhealthy_without_raising(self) -> None:
        fake = FakeAnthropicClient()

        async def failing_count_tokens(**_: object) -> None:
            raise RuntimeError("provider down")

        fake.messages.count_tokens = failing_count_tokens  # type: ignore[assignment]
        provider = ClaudeProvider(ClaudeSettings(api_key=SecretStr("k")), client=fake)

        health = await provider.health_check()

        assert health.status.is_serving is False
        assert "provider down" in (health.message or "")

    async def test_aclose_closes_the_client(self) -> None:
        provider, fake = make_provider()
        await provider.aclose()

        assert fake.closed is True

    async def test_async_context_manager_closes_on_exit(self) -> None:
        provider, fake = make_provider()
        async with provider:
            pass

        assert fake.closed is True
