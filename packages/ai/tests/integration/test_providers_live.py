"""Integration tests against live AI providers.

Ollama runs in the dev compose stack, so these are the tests that prove the
abstraction works against a real model server rather than a fixture. The Claude
tests only run when ``FIE_CLAUDE_API_KEY`` is set, so a normal CI run costs
nothing; they exist so the primary production provider is exercised for real
before a release.
"""

from __future__ import annotations

import pytest

from fie_ai.config import ClaudeSettings, OllamaSettings
from fie_ai.contracts import CompletionRequest, Effort, Message, Role, ThinkingMode
from fie_ai.providers.ollama import OllamaProvider
from fie_ai.structured import parse_structured, structured_request
from fie_schemas.base import FIEModel
from fie_testing.infra import (
    ollama_endpoint,
    require_anthropic_key,
    require_ollama_model,
    require_service,
)

pytestmark = [pytest.mark.integration]


class Classification(FIEModel):
    sentiment: str
    confidence: float


def request(prompt: str, **kwargs: object) -> CompletionRequest:
    return CompletionRequest(
        messages=[Message(role=Role.USER, content=prompt)],
        **kwargs,  # type: ignore[arg-type]
    )


class TestOllamaLive:
    @pytest.fixture
    async def provider(self):  # type: ignore[no-untyped-def]
        require_service(ollama_endpoint())
        settings = OllamaSettings()
        # A reachable server without the model answers 404, which is otherwise
        # indistinguishable from a provider bug.
        require_ollama_model(settings.model)
        instance = OllamaProvider(settings)
        yield instance
        await instance.aclose()

    async def test_health_check_against_a_running_server(self, provider) -> None:  # type: ignore[no-untyped-def]
        assert (await provider.health_check()).status.is_serving is True

    async def test_completes_a_prompt(self, provider) -> None:  # type: ignore[no-untyped-def]
        response = await provider.complete(
            request("Reply with exactly the word: ready", max_tokens=32)
        )

        assert response.text.strip()
        assert response.provider == "ollama"
        assert response.usage.output_tokens > 0

    async def test_streams_tokens(self, provider) -> None:  # type: ignore[no-untyped-def]
        chunks = [
            chunk async for chunk in provider.stream(request("Count to three.", max_tokens=48))
        ]

        assert "".join(chunks).strip()

    async def test_system_prompt_is_honored(self, provider) -> None:  # type: ignore[no-untyped-def]
        response = await provider.complete(
            request(
                "What are you?",
                system="You always answer in exactly one word.",
                max_tokens=32,
            )
        )

        assert response.text.strip()


class TestClaudeLive:
    """Only runs when a real API key is configured."""

    @pytest.fixture
    async def provider(self):  # type: ignore[no-untyped-def]
        require_anthropic_key()
        from fie_ai.providers.claude import ClaudeProvider

        instance = ClaudeProvider(ClaudeSettings())
        yield instance
        await instance.aclose()

    async def test_health_check_uses_the_free_endpoint(self, provider) -> None:  # type: ignore[no-untyped-def]
        assert (await provider.health_check()).status.is_serving is True

    async def test_token_counting_returns_a_real_count(self, provider) -> None:  # type: ignore[no-untyped-def]
        count = await provider.count_tokens(request("How many tokens is this?"))

        assert count > 0

    async def test_adaptive_thinking_completion(self, provider) -> None:  # type: ignore[no-untyped-def]
        response = await provider.complete(
            request(
                "Reply with exactly the word: ready",
                max_tokens=64,
                thinking=ThinkingMode.ADAPTIVE,
                effort=Effort.LOW,
            )
        )

        assert response.text.strip()
        assert response.usage.input_tokens > 0

    async def test_structured_output_validates(self, provider) -> None:  # type: ignore[no-untyped-def]
        # Structured output is what makes AI behaviour assertable without
        # matching on exact model prose.
        structured = structured_request(
            request(
                "Classify the sentiment of: 'Revenue beat expectations by 12%.'",
                max_tokens=256,
            ),
            Classification,
        )

        result = parse_structured(await provider.complete(structured), Classification)

        assert result.sentiment
        assert 0.0 <= result.confidence <= 1.0

    async def test_sampling_parameters_are_dropped_not_rejected(self, provider) -> None:  # type: ignore[no-untyped-def]
        # The neutral contract allows temperature; current Claude models reject
        # it with a 400. The provider must absorb that, so this must not raise.
        response = await provider.complete(
            request("Reply with exactly the word: ready", max_tokens=32, temperature=0.5)
        )

        assert response.text.strip()
