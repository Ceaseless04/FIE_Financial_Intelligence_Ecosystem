"""Unit tests for the provider-neutral AI contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fie_ai.contracts import (
    CompletionRequest,
    CompletionResponse,
    Effort,
    Message,
    ProviderCapabilities,
    Role,
    StopReason,
    ThinkingMode,
    TokenUsage,
)

pytestmark = pytest.mark.unit


class TestMessage:
    def test_requires_non_empty_content(self) -> None:
        with pytest.raises(ValidationError):
            Message(role=Role.USER, content="")

    def test_is_immutable(self) -> None:
        message = Message(role=Role.USER, content="hello")

        with pytest.raises(ValidationError):
            message.content = "changed"  # type: ignore[misc]


class TestCompletionRequest:
    def test_minimal_request_uses_platform_defaults(self) -> None:
        request = CompletionRequest(messages=[Message(role=Role.USER, content="hi")])

        assert request.thinking is ThinkingMode.ADAPTIVE
        assert request.cache_system_prompt is True
        assert request.model is None
        assert request.max_tokens == 4096

    def test_requires_at_least_one_message(self) -> None:
        with pytest.raises(ValidationError):
            CompletionRequest(messages=[])

    def test_conversation_must_start_with_a_user_message(self) -> None:
        with pytest.raises(ValidationError, match="must start with a user message"):
            CompletionRequest(messages=[Message(role=Role.ASSISTANT, content="hi")])

    def test_max_tokens_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            CompletionRequest(messages=[Message(role=Role.USER, content="hi")], max_tokens=200_000)
        with pytest.raises(ValidationError):
            CompletionRequest(messages=[Message(role=Role.USER, content="hi")], max_tokens=0)

    def test_temperature_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            CompletionRequest(messages=[Message(role=Role.USER, content="hi")], temperature=1.5)

    @pytest.mark.parametrize(
        ("max_tokens", "expected"), [(4096, False), (16_000, False), (16_001, True), (64_000, True)]
    )
    def test_large_outputs_require_streaming(self, max_tokens: int, expected: bool) -> None:
        # Non-streaming requests above ~16K risk an HTTP timeout.
        request = CompletionRequest(
            messages=[Message(role=Role.USER, content="hi")], max_tokens=max_tokens
        )

        assert request.requires_streaming is expected

    def test_response_schema_must_be_an_object(self) -> None:
        with pytest.raises(ValidationError):
            CompletionRequest(
                messages=[Message(role=Role.USER, content="hi")],
                response_schema=["not", "an", "object"],  # type: ignore[arg-type]
            )

    def test_model_copy_supports_derived_requests(self) -> None:
        base = CompletionRequest(messages=[Message(role=Role.USER, content="hi")])
        derived = base.model_copy(update={"effort": Effort.HIGH})

        assert derived.effort is Effort.HIGH
        assert base.effort is None


class TestCompletionResponse:
    def test_defaults_are_a_successful_turn(self) -> None:
        response = CompletionResponse(text="hi", model="m", provider="p")

        assert response.stop_reason is StopReason.END_TURN
        assert response.is_refusal is False
        assert response.is_truncated is False
        assert response.usage.total_tokens == 0

    def test_refusal_is_detected(self) -> None:
        response = CompletionResponse(
            text="",
            model="m",
            provider="p",
            stop_reason=StopReason.REFUSAL,
            refusal_category="cyber",
        )

        assert response.is_refusal is True
        assert response.refusal_category == "cyber"

    def test_truncation_is_detected(self) -> None:
        response = CompletionResponse(
            text="partial", model="m", provider="p", stop_reason=StopReason.MAX_TOKENS
        )

        assert response.is_truncated is True

    def test_usage_is_carried(self) -> None:
        response = CompletionResponse(
            text="hi",
            model="m",
            provider="p",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

        assert response.usage.total_tokens == 15

    def test_served_by_records_a_fallback(self) -> None:
        response = CompletionResponse(text="hi", model="m", provider="p", served_by="ollama")

        assert response.served_by == "ollama"


class TestProviderCapabilities:
    def test_defaults_are_conservative(self) -> None:
        capabilities = ProviderCapabilities()

        assert capabilities.supports_structured_output is False
        assert capabilities.supports_thinking is False
        assert capabilities.supports_streaming is True

    def test_sampling_support_is_expressible(self) -> None:
        # Current Claude model families reject temperature outright, so the
        # abstraction has to be able to say so.
        capabilities = ProviderCapabilities(supports_sampling_params=False)

        assert capabilities.supports_sampling_params is False
