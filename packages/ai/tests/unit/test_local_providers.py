"""Unit tests for the Ollama and vLLM providers.

HTTP is intercepted at the transport layer with respx, so the providers'
request construction and response parsing run for real against realistic
payloads.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from fie_ai.config import OllamaSettings, VLLMSettings
from fie_ai.contracts import CompletionRequest, Message, Role, StopReason
from fie_ai.providers.ollama import OllamaProvider
from fie_ai.providers.vllm import VLLMProvider
from fie_common.config import SecretStr
from fie_common.errors import (
    AIProviderError,
    AIProviderRateLimitError,
    AIProviderTimeoutError,
    AuthenticationError,
    ValidationError,
)
from fie_common.resilience import ResiliencePolicy, RetryPolicy

pytestmark = pytest.mark.unit

OLLAMA_BASE = "http://ollama.test:11434"
VLLM_BASE = "http://vllm.test:8000/v1"


def no_retry() -> ResiliencePolicy:
    return ResiliencePolicy(timeout_seconds=5.0, retry=RetryPolicy(max_attempts=1), breaker=None)


def request(prompt: str = "Extract the revenue figure.", **kwargs: object) -> CompletionRequest:
    return CompletionRequest(
        messages=[Message(role=Role.USER, content=prompt)],
        **kwargs,  # type: ignore[arg-type]
    )


def ollama_provider() -> OllamaProvider:
    return OllamaProvider(
        OllamaSettings(base_url=OLLAMA_BASE, model="llama3.1:8b"), resilience=no_retry()
    )


def vllm_provider(api_key: str | None = None) -> VLLMProvider:
    return VLLMProvider(
        VLLMSettings(
            base_url=VLLM_BASE,
            model="meta-llama/Llama-3.1-8B-Instruct",
            api_key=SecretStr(api_key) if api_key else None,
        ),
        resilience=no_retry(),
    )


OLLAMA_OK = {
    "model": "llama3.1:8b",
    "message": {"role": "assistant", "content": "Revenue was $1.2B."},
    "done_reason": "stop",
    "prompt_eval_count": 30,
    "eval_count": 12,
}

VLLM_OK = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "choices": [
        {"message": {"role": "assistant", "content": "Revenue was $1.2B."}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 30, "completion_tokens": 12},
}


class TestOllamaProvider:
    @respx.mock
    async def test_parses_a_successful_response(self) -> None:
        respx.post(f"{OLLAMA_BASE}/api/chat").mock(return_value=httpx.Response(200, json=OLLAMA_OK))
        provider = ollama_provider()

        response = await provider.complete(request())

        assert response.text == "Revenue was $1.2B."
        assert response.provider == "ollama"
        assert response.usage.input_tokens == 30
        assert response.usage.output_tokens == 12
        await provider.aclose()

    @respx.mock
    async def test_system_prompt_becomes_a_system_message(self) -> None:
        route = respx.post(f"{OLLAMA_BASE}/api/chat").mock(
            return_value=httpx.Response(200, json=OLLAMA_OK)
        )
        provider = ollama_provider()

        await provider.complete(request(system="You are a filing analyst."))

        payload = route.calls.last.request.read()
        assert b'"role": "system"' in payload or b'"role":"system"' in payload
        await provider.aclose()

    @respx.mock
    async def test_max_tokens_maps_to_num_predict(self) -> None:
        route = respx.post(f"{OLLAMA_BASE}/api/chat").mock(
            return_value=httpx.Response(200, json=OLLAMA_OK)
        )
        provider = ollama_provider()

        await provider.complete(request(max_tokens=256))

        assert b"num_predict" in route.calls.last.request.read()
        await provider.aclose()

    @respx.mock
    async def test_temperature_is_forwarded(self) -> None:
        route = respx.post(f"{OLLAMA_BASE}/api/chat").mock(
            return_value=httpx.Response(200, json=OLLAMA_OK)
        )
        provider = ollama_provider()

        await provider.complete(request(temperature=0.2))

        assert b"temperature" in route.calls.last.request.read()
        await provider.aclose()

    @respx.mock
    async def test_response_schema_is_sent_as_format(self) -> None:
        route = respx.post(f"{OLLAMA_BASE}/api/chat").mock(
            return_value=httpx.Response(200, json=OLLAMA_OK)
        )
        provider = ollama_provider()

        await provider.complete(request(response_schema={"type": "object"}))

        assert b"format" in route.calls.last.request.read()
        await provider.aclose()

    @respx.mock
    async def test_length_stop_reason_maps_to_max_tokens(self) -> None:
        respx.post(f"{OLLAMA_BASE}/api/chat").mock(
            return_value=httpx.Response(200, json={**OLLAMA_OK, "done_reason": "length"})
        )
        provider = ollama_provider()

        response = await provider.complete(request())

        assert response.stop_reason is StopReason.MAX_TOKENS
        await provider.aclose()

    @respx.mock
    async def test_bad_request_maps_to_validation_error(self) -> None:
        respx.post(f"{OLLAMA_BASE}/api/chat").mock(return_value=httpx.Response(400))
        provider = ollama_provider()

        with pytest.raises(ValidationError):
            await provider.complete(request())
        await provider.aclose()

    @respx.mock
    async def test_server_error_is_retryable(self) -> None:
        respx.post(f"{OLLAMA_BASE}/api/chat").mock(return_value=httpx.Response(500))
        provider = ollama_provider()

        with pytest.raises(AIProviderError) as exc_info:
            await provider.complete(request())

        assert exc_info.value.retryable is True
        await provider.aclose()

    @respx.mock
    async def test_timeout_maps_to_a_provider_timeout(self) -> None:
        respx.post(f"{OLLAMA_BASE}/api/chat").mock(side_effect=httpx.ConnectTimeout("timed out"))
        provider = ollama_provider()

        with pytest.raises(AIProviderTimeoutError):
            await provider.complete(request())
        await provider.aclose()

    @respx.mock
    async def test_connection_failure_is_retryable(self) -> None:
        respx.post(f"{OLLAMA_BASE}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
        provider = ollama_provider()

        with pytest.raises(AIProviderError):
            await provider.complete(request())
        await provider.aclose()

    @respx.mock
    async def test_streaming_yields_ndjson_deltas(self) -> None:
        body = (
            '{"message":{"content":"Revenue "}}\n'
            '{"message":{"content":"was "}}\n'
            '{"message":{"content":"$1.2B."}}\n'
        )
        respx.post(f"{OLLAMA_BASE}/api/chat").mock(return_value=httpx.Response(200, text=body))
        provider = ollama_provider()

        chunks = [chunk async for chunk in provider.stream(request())]

        assert "".join(chunks) == "Revenue was $1.2B."
        await provider.aclose()

    @respx.mock
    async def test_health_check_probes_the_tags_endpoint(self) -> None:
        respx.get(f"{OLLAMA_BASE}/api/tags").mock(
            return_value=httpx.Response(200, json={"models": []})
        )
        provider = ollama_provider()

        health = await provider.health_check()

        assert health.status.is_serving is True
        await provider.aclose()

    @respx.mock
    async def test_health_check_reports_unhealthy_when_down(self) -> None:
        respx.get(f"{OLLAMA_BASE}/api/tags").mock(side_effect=httpx.ConnectError("refused"))
        provider = ollama_provider()

        health = await provider.health_check()

        assert health.status.is_serving is False
        await provider.aclose()

    def test_capabilities_reflect_local_model_limits(self) -> None:
        capabilities = OllamaProvider(OllamaSettings(base_url=OLLAMA_BASE)).capabilities

        assert capabilities.supports_thinking is False
        assert capabilities.supports_token_counting is False
        assert capabilities.supports_sampling_params is True

    async def test_token_counting_falls_back_to_an_estimate(self) -> None:
        provider = ollama_provider()

        count = await provider.count_tokens(request("a" * 400))

        assert count > 0
        await provider.aclose()


class TestVLLMProvider:
    @respx.mock
    async def test_parses_an_openai_compatible_response(self) -> None:
        respx.post(f"{VLLM_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=VLLM_OK)
        )
        provider = vllm_provider()

        response = await provider.complete(request())

        assert response.text == "Revenue was $1.2B."
        assert response.provider == "vllm"
        assert response.usage.input_tokens == 30
        await provider.aclose()

    @respx.mock
    async def test_api_key_is_sent_as_a_bearer_token(self) -> None:
        route = respx.post(f"{VLLM_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=VLLM_OK)
        )
        provider = vllm_provider(api_key="vllm-secret")

        await provider.complete(request())

        assert route.calls.last.request.headers["authorization"] == "Bearer vllm-secret"
        await provider.aclose()

    @respx.mock
    async def test_response_schema_uses_guided_decoding(self) -> None:
        route = respx.post(f"{VLLM_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=VLLM_OK)
        )
        provider = vllm_provider()

        await provider.complete(request(response_schema={"type": "object"}))

        assert b"guided_json" in route.calls.last.request.read()
        await provider.aclose()

    @respx.mock
    async def test_empty_choices_is_an_error(self) -> None:
        respx.post(f"{VLLM_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )
        provider = vllm_provider()

        with pytest.raises(AIProviderError, match="no choices"):
            await provider.complete(request())
        await provider.aclose()

    @pytest.mark.parametrize(
        ("finish_reason", "expected"),
        [
            ("stop", StopReason.END_TURN),
            ("length", StopReason.MAX_TOKENS),
            ("tool_calls", StopReason.TOOL_USE),
            ("content_filter", StopReason.REFUSAL),
        ],
    )
    @respx.mock
    async def test_finish_reason_mapping(self, finish_reason: str, expected: StopReason) -> None:
        payload = {
            **VLLM_OK,
            "choices": [{"message": {"content": "text"}, "finish_reason": finish_reason}],
        }
        respx.post(f"{VLLM_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=payload)
        )
        provider = vllm_provider()

        response = await provider.complete(request())

        assert response.stop_reason is expected
        await provider.aclose()

    @pytest.mark.parametrize("status", [401, 403])
    @respx.mock
    async def test_auth_failures_map_to_authentication_error(self, status: int) -> None:
        respx.post(f"{VLLM_BASE}/chat/completions").mock(return_value=httpx.Response(status))
        provider = vllm_provider()

        with pytest.raises(AuthenticationError):
            await provider.complete(request())
        await provider.aclose()

    @respx.mock
    async def test_rate_limit_maps_to_a_retryable_error(self) -> None:
        respx.post(f"{VLLM_BASE}/chat/completions").mock(return_value=httpx.Response(429))
        provider = vllm_provider()

        with pytest.raises(AIProviderRateLimitError) as exc_info:
            await provider.complete(request())

        assert exc_info.value.retryable is True
        await provider.aclose()

    @respx.mock
    async def test_bad_request_maps_to_validation_error(self) -> None:
        respx.post(f"{VLLM_BASE}/chat/completions").mock(return_value=httpx.Response(400))
        provider = vllm_provider()

        with pytest.raises(ValidationError):
            await provider.complete(request())
        await provider.aclose()

    @respx.mock
    async def test_streaming_parses_sse_deltas(self) -> None:
        body = (
            'data: {"choices":[{"delta":{"content":"Revenue "}}]}\n'
            'data: {"choices":[{"delta":{"content":"was "}}]}\n'
            'data: {"choices":[{"delta":{"content":"$1.2B."}}]}\n'
            "data: [DONE]\n"
        )
        respx.post(f"{VLLM_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, text=body)
        )
        provider = vllm_provider()

        chunks = [chunk async for chunk in provider.stream(request())]

        assert "".join(chunks) == "Revenue was $1.2B."
        await provider.aclose()

    @respx.mock
    async def test_health_check_probes_the_models_endpoint(self) -> None:
        respx.get(f"{VLLM_BASE}/models").mock(return_value=httpx.Response(200, json={"data": []}))
        provider = vllm_provider()

        health = await provider.health_check()

        assert health.status.is_serving is True
        await provider.aclose()
