"""Ollama provider — local models for development, evaluation, and cost control.

Ollama runs in the dev compose stack so the ecosystem is fully exercisable
without an API key or network egress. It is also the substrate for offline
evaluation runs where determinism matters more than capability.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import httpx

from fie_ai.base import AIProvider
from fie_ai.config import OllamaSettings
from fie_ai.contracts import (
    CompletionRequest,
    CompletionResponse,
    ProviderCapabilities,
    StopReason,
)
from fie_ai.http import build_async_client
from fie_common.errors import AIProviderError, AIProviderTimeoutError, ValidationError
from fie_common.resilience import ResiliencePolicy
from fie_observability.metrics import TokenUsage


class OllamaProvider(AIProvider):
    """Local models served by Ollama's native chat API."""

    name: ClassVar[str] = "ollama"

    def __init__(
        self,
        settings: OllamaSettings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        resilience: ResiliencePolicy | None = None,
    ) -> None:
        super().__init__(resilience=resilience)
        self._settings = settings or OllamaSettings()
        self._client = client or build_async_client(
            base_url=self._settings.base_url,
            timeout=self._settings.timeout_seconds,
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            # Recent Ollama accepts a JSON Schema in `format`; older builds only
            # accept "json". Treated as best-effort and validated after the fact.
            supports_structured_output=True,
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

        options: dict[str, Any] = {"num_predict": request.max_tokens}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.stop_sequences:
            options["stop"] = list(request.stop_sequences)

        payload: dict[str, Any] = {
            "model": self.resolve_model(request),
            "messages": messages,
            "stream": stream,
            "options": options,
        }
        if request.response_schema is not None:
            payload["format"] = request.response_schema
        return payload

    async def _complete(self, request: CompletionRequest) -> CompletionResponse:
        payload = self._build_payload(request, stream=False)
        data = await self._post("/api/chat", payload)

        message = data.get("message") or {}
        return CompletionResponse(
            text=message.get("content", ""),
            model=data.get("model", payload["model"]),
            provider=self.name,
            stop_reason=self._to_stop_reason(data.get("done_reason")),
            usage=TokenUsage(
                input_tokens=int(data.get("prompt_eval_count") or 0),
                output_tokens=int(data.get("eval_count") or 0),
            ),
        )

    async def _stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        payload = self._build_payload(request, stream=True)
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    content = (chunk.get("message") or {}).get("content")
                    if content:
                        yield content
        except httpx.TimeoutException as error:
            raise AIProviderTimeoutError(
                f"Ollama request timed out: {error}", details={"provider": self.name}
            ) from error
        except httpx.HTTPError as error:
            raise AIProviderError(
                f"Ollama stream failed: {error}", details={"provider": self.name}
            ) from error

    async def _ping(self) -> None:
        response = await self._client.get("/api/tags")
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
                f"Ollama request timed out: {error}", details={"provider": self.name}
            ) from error
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            if status == 400:
                raise ValidationError(
                    f"Ollama rejected the request: {error}",
                    details={"provider": self.name, "status_code": status},
                ) from error
            raise AIProviderError(
                f"Ollama returned HTTP {status}",
                details={"provider": self.name, "status_code": status},
            ) from error
        except httpx.HTTPError as error:
            raise AIProviderError(
                f"Ollama connection error: {error}", details={"provider": self.name}
            ) from error

    @staticmethod
    def _to_stop_reason(raw: str | None) -> StopReason:
        return {
            "stop": StopReason.END_TURN,
            "length": StopReason.MAX_TOKENS,
        }.get(raw or "stop", StopReason.END_TURN)


__all__ = ["OllamaProvider"]
