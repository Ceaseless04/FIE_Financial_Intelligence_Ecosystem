"""vLLM provider — self-hosted inference over the OpenAI-compatible API.

Used for high-throughput batch workloads (bulk filing extraction, embedding
pipelines, evaluation sweeps) where per-token API pricing dominates and the
task does not need frontier reasoning.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import httpx

from fie_ai.base import AIProvider
from fie_ai.config import VLLMSettings
from fie_ai.contracts import (
    CompletionRequest,
    CompletionResponse,
    ProviderCapabilities,
    StopReason,
)
from fie_ai.http import build_async_client
from fie_common.config import secret_value
from fie_common.errors import (
    AIProviderError,
    AIProviderRateLimitError,
    AIProviderTimeoutError,
    AuthenticationError,
    ValidationError,
)
from fie_common.resilience import ResiliencePolicy
from fie_observability.metrics import TokenUsage


class VLLMProvider(AIProvider):
    """Self-hosted models behind vLLM's OpenAI-compatible server."""

    name: ClassVar[str] = "vllm"

    def __init__(
        self,
        settings: VLLMSettings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        resilience: ResiliencePolicy | None = None,
    ) -> None:
        super().__init__(resilience=resilience)
        self._settings = settings or VLLMSettings()

        if client is not None:
            self._client = client
        else:
            headers: dict[str, str] = {}
            api_key = secret_value(self._settings.api_key)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            self._client = build_async_client(
                base_url=self._settings.base_url,
                timeout=self._settings.timeout_seconds,
                headers=headers,
            )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_structured_output=True,  # via guided decoding
            supports_thinking=False,
            supports_effort=False,
            supports_sampling_params=True,
            supports_token_counting=False,
            supports_prompt_caching=False,
            max_context_tokens=128_000,
        )

    @property
    def default_model(self) -> str:
        return self._settings.model

    def _build_payload(self, request: CompletionRequest, *, stream: bool) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(
            {"role": str(message.role), "content": message.content} for message in request.messages
        )

        payload: dict[str, Any] = {
            "model": self.resolve_model(request),
            "messages": messages,
            "max_tokens": request.max_tokens,
            "stream": stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.stop_sequences:
            payload["stop"] = list(request.stop_sequences)
        if request.response_schema is not None:
            # vLLM guided decoding constrains generation to the schema.
            payload["guided_json"] = request.response_schema
        return payload

    async def _complete(self, request: CompletionRequest) -> CompletionResponse:
        payload = self._build_payload(request, stream=False)
        data = await self._post("/chat/completions", payload)

        choices = data.get("choices") or []
        if not choices:
            raise AIProviderError("vLLM returned no choices", details={"provider": self.name})
        choice = choices[0]
        usage = data.get("usage") or {}

        return CompletionResponse(
            text=(choice.get("message") or {}).get("content", ""),
            model=data.get("model", payload["model"]),
            provider=self.name,
            stop_reason=self._to_stop_reason(choice.get("finish_reason")),
            usage=TokenUsage(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
            ),
        )

    async def _stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        payload = self._build_payload(request, stream=True)
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    body = line.removeprefix("data:").strip()
                    if not body or body == "[DONE]":
                        continue
                    chunk = json.loads(body)
                    for choice in chunk.get("choices") or []:
                        content = (choice.get("delta") or {}).get("content")
                        if content:
                            yield content
        except httpx.TimeoutException as error:
            raise AIProviderTimeoutError(
                f"vLLM request timed out: {error}", details={"provider": self.name}
            ) from error
        except httpx.HTTPError as error:
            raise AIProviderError(
                f"vLLM stream failed: {error}", details={"provider": self.name}
            ) from error

    async def _ping(self) -> None:
        response = await self._client.get("/models")
        response.raise_for_status()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=payload)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            return data
        except httpx.TimeoutException as error:
            raise AIProviderTimeoutError(
                f"vLLM request timed out: {error}", details={"provider": self.name}
            ) from error
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            details = {"provider": self.name, "status_code": status}
            if status in (401, 403):
                raise AuthenticationError(
                    f"vLLM authentication failed (HTTP {status})", details=details
                ) from error
            if status == 429:
                raise AIProviderRateLimitError(
                    "vLLM rate limit exceeded", details=details
                ) from error
            if status == 400:
                raise ValidationError(
                    f"vLLM rejected the request: {error}", details=details
                ) from error
            raise AIProviderError(f"vLLM returned HTTP {status}", details=details) from error
        except httpx.HTTPError as error:
            raise AIProviderError(
                f"vLLM connection error: {error}", details={"provider": self.name}
            ) from error

    @staticmethod
    def _to_stop_reason(raw: str | None) -> StopReason:
        return {
            "stop": StopReason.END_TURN,
            "length": StopReason.MAX_TOKENS,
            "tool_calls": StopReason.TOOL_USE,
            "content_filter": StopReason.REFUSAL,
        }.get(raw or "stop", StopReason.END_TURN)


__all__ = ["VLLMProvider"]
