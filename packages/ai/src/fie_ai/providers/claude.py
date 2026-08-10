"""Anthropic Claude provider — the default production reasoning model.

This is the only module in the ecosystem that imports the ``anthropic`` SDK.
Everything above it speaks ``fie_ai.contracts``.

Three current-API details this provider is responsible for absorbing so that
callers never have to know them:

* Sampling parameters (``temperature``/``top_p``/``top_k``) are rejected with a
  400 by the current model families. A caller may set ``temperature`` on a
  provider-neutral request; we drop it here rather than fail the call.
* ``thinking`` is adaptive — ``budget_tokens`` is gone. Disabling thinking is
  only legal at effort ``high`` or below.
* A refusal arrives as HTTP 200 with ``stop_reason == "refusal"``, so
  ``stop_reason`` is checked before the content array is ever indexed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, ClassVar

import anthropic

from fie_ai.base import AIProvider
from fie_ai.config import ClaudeSettings
from fie_ai.contracts import (
    CompletionRequest,
    CompletionResponse,
    Effort,
    ProviderCapabilities,
    StopReason,
    ThinkingMode,
)
from fie_common.config import Environment, secret_value
from fie_common.errors import (
    AIProviderError,
    AIProviderRateLimitError,
    AIProviderTimeoutError,
    AuthenticationError,
    ConfigurationError,
    ValidationError,
)
from fie_common.resilience import ResiliencePolicy
from fie_observability.metrics import TokenUsage

#: Effort levels that cannot be combined with disabled thinking.
_EFFORT_REQUIRING_THINKING = frozenset({Effort.XHIGH, Effort.MAX})

#: Model families that reject temperature/top_p/top_k with a 400.
_NO_SAMPLING_PREFIXES = ("claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-5")


class ClaudeProvider(AIProvider):
    """Claude via the official Anthropic SDK."""

    name: ClassVar[str] = "claude"

    def __init__(
        self,
        settings: ClaudeSettings | None = None,
        *,
        client: Any | None = None,
        resilience: ResiliencePolicy | None = None,
    ) -> None:
        super().__init__(resilience=resilience)
        self._settings = settings or ClaudeSettings()

        if client is not None:
            self._client = client
        else:
            api_key = secret_value(self._settings.api_key)
            if not api_key:
                raise ConfigurationError(
                    "FIE_CLAUDE_API_KEY is not set; cannot construct the Claude provider",
                    details={"provider": self.name},
                )
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": self._settings.timeout_seconds,
                "max_retries": self._settings.max_retries,
            }
            if self._settings.base_url:
                kwargs["base_url"] = self._settings.base_url
            self._client = anthropic.AsyncAnthropic(**kwargs)

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_structured_output=True,
            supports_thinking=True,
            supports_effort=True,
            supports_sampling_params=False,
            supports_token_counting=True,
            supports_prompt_caching=True,
            max_context_tokens=1_000_000,
        )

    @property
    def default_model(self) -> str:
        return self._settings.model

    def validate_for(self, environment: Environment) -> None:
        self._settings.validate_for(environment)

    # -- request construction ------------------------------------------------

    def _build_payload(self, request: CompletionRequest) -> dict[str, Any]:
        model = self.resolve_model(request)

        if (
            request.thinking is ThinkingMode.DISABLED
            and request.effort in _EFFORT_REQUIRING_THINKING
        ):
            raise ValidationError(
                "thinking cannot be disabled at effort 'xhigh' or 'max'",
                details={"model": model, "effort": str(request.effort)},
            )

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": [
                {"role": str(message.role), "content": message.content}
                for message in request.messages
            ],
            "thinking": {"type": str(request.thinking)},
        }

        if request.system:
            # A cache breakpoint on the system block caches tools + system
            # together; a short prefix simply won't cache, which is harmless.
            if request.cache_system_prompt:
                payload["system"] = [
                    {
                        "type": "text",
                        "text": request.system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                payload["system"] = request.system

        output_config: dict[str, Any] = {}
        if request.effort is not None:
            output_config["effort"] = str(request.effort)
        if request.response_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": request.response_schema,
            }
        if output_config:
            payload["output_config"] = output_config

        if request.stop_sequences:
            payload["stop_sequences"] = list(request.stop_sequences)

        # Sampling parameters are rejected outright by current model families.
        # Dropping rather than forwarding keeps the neutral contract usable
        # across providers that do accept them.
        if request.temperature is not None and self._model_accepts_sampling(model):
            payload["temperature"] = request.temperature

        return payload

    @staticmethod
    def _model_accepts_sampling(model: str) -> bool:
        return not model.startswith(_NO_SAMPLING_PREFIXES)

    # -- provider hooks ------------------------------------------------------

    async def _complete(self, request: CompletionRequest) -> CompletionResponse:
        payload = self._build_payload(request)
        try:
            if request.requires_streaming:
                # Non-streaming requests above ~16K max_tokens risk an HTTP
                # timeout, so large outputs always stream and are reassembled.
                async with self._client.messages.stream(**payload) as stream:
                    message = await stream.get_final_message()
            else:
                message = await self._client.messages.create(**payload)
        except Exception as error:
            raise self._translate_error(error) from error

        return self._to_response(message, payload["model"])

    async def _stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        payload = self._build_payload(request)
        try:
            async with self._client.messages.stream(**payload) as stream:
                async for chunk in stream.text_stream:
                    yield chunk
        except Exception as error:
            raise self._translate_error(error) from error

    async def _ping(self) -> None:
        """Token counting is free and exercises auth plus connectivity."""
        await self._client.messages.count_tokens(
            model=self.default_model,
            messages=[{"role": "user", "content": "ping"}],
        )

    async def count_tokens(self, request: CompletionRequest) -> int:
        payload: dict[str, Any] = {
            "model": self.resolve_model(request),
            "messages": [
                {"role": str(message.role), "content": message.content}
                for message in request.messages
            ],
        }
        if request.system:
            payload["system"] = request.system
        try:
            result = await self._client.messages.count_tokens(**payload)
        except Exception as error:
            raise self._translate_error(error) from error
        tokens: int = result.input_tokens
        return tokens

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()

    # -- response translation ------------------------------------------------

    def _to_response(self, message: Any, model: str) -> CompletionResponse:
        stop_reason = self._to_stop_reason(getattr(message, "stop_reason", None))

        # Check stop_reason before touching content: a refusal returns HTTP 200
        # with an empty (pre-output) or partial (mid-stream) content array.
        refusal_category: str | None = None
        if stop_reason is StopReason.REFUSAL:
            details = getattr(message, "stop_details", None)
            refusal_category = getattr(details, "category", None)

        text = self._extract_text(message)
        return CompletionResponse(
            text=text,
            model=getattr(message, "model", model),
            provider=self.name,
            stop_reason=stop_reason,
            usage=self._to_usage(getattr(message, "usage", None)),
            refusal_category=refusal_category,
        )

    @staticmethod
    def _extract_text(message: Any) -> str:
        """Concatenate text blocks, skipping thinking and tool-use blocks."""
        blocks = getattr(message, "content", None) or []
        parts: list[str] = []
        for block in blocks:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        return "".join(parts)

    @staticmethod
    def _to_usage(usage: Any) -> TokenUsage:
        if usage is None:
            return TokenUsage()
        return TokenUsage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )

    @staticmethod
    def _to_stop_reason(raw: str | None) -> StopReason:
        if raw is None:
            return StopReason.END_TURN
        try:
            return StopReason(raw)
        except ValueError:
            return StopReason.END_TURN

    # -- error translation ---------------------------------------------------

    def _translate_error(self, error: Exception) -> Exception:
        """Map SDK exceptions onto the platform error hierarchy.

        Classification drives retry: rate limits and connection failures are
        retryable, a bad request or bad key is not.
        """
        details = {"provider": self.name}

        if isinstance(error, anthropic.APITimeoutError):
            return AIProviderTimeoutError(f"Claude request timed out: {error}", details=details)
        if isinstance(error, anthropic.RateLimitError):
            return AIProviderRateLimitError(
                f"Claude rate limit exceeded: {error}",
                details={**details, "retry_after": self._retry_after(error)},
            )
        if isinstance(error, anthropic.AuthenticationError):
            return AuthenticationError(f"Claude authentication failed: {error}", details=details)
        if isinstance(error, anthropic.BadRequestError):
            return ValidationError(f"Claude rejected the request: {error}", details=details)
        if isinstance(error, anthropic.APIConnectionError):
            return AIProviderError(f"Claude connection error: {error}", details=details)
        if isinstance(error, anthropic.APIStatusError):
            return AIProviderError(
                f"Claude API error ({error.status_code}): {error}",
                details={**details, "status_code": error.status_code},
            )
        if isinstance(error, (ValidationError, AIProviderError, AuthenticationError)):
            return error
        return AIProviderError(f"Claude call failed: {error}", details=details)

    @staticmethod
    def _retry_after(error: Exception) -> float | None:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            return None
        raw = headers.get("retry-after")
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None


__all__ = ["ClaudeProvider"]
