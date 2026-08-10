"""Embedding provider abstraction.

Companion to :mod:`fie_ai.base`: completions reason, embeddings retrieve. Both
sit behind an abstraction so application code never imports a vendor SDK.

Anthropic does not expose an embeddings endpoint, so embeddings come from a
local model server (Ollama in the dev stack, vLLM or a managed endpoint in
production). That asymmetry is why this is a separate interface rather than
another method on ``AIProvider``.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

import httpx

from fie_ai.http import build_async_client
from fie_common.errors import (
    AIProviderError,
    AIProviderTimeoutError,
    ValidationError,
)
from fie_common.resilience import ResiliencePolicy, RetryPolicy
from fie_observability.logging import get_logger
from fie_schemas.health import ComponentHealth, HealthStatus

logger = get_logger(__name__)

#: A single embedding vector.
Vector = list[float]


class EmbeddingProvider(ABC):
    """Turns text into vectors for similarity retrieval."""

    name: ClassVar[str] = "unknown"

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector width. Must match the storage column, or writes will fail."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier, recorded alongside stored vectors."""

    @abstractmethod
    async def embed_texts(self, texts: Sequence[str]) -> list[Vector]:
        """Embed a batch of texts, preserving input order."""

    async def embed_text(self, text: str) -> Vector:
        """Embed a single text."""
        vectors = await self.embed_texts([text])
        return vectors[0]

    async def health_check(self) -> ComponentHealth:
        started = time.perf_counter()
        try:
            await self.embed_text("health check")
        except Exception as error:  # noqa: BLE001 — health must never raise
            return ComponentHealth(
                name=f"embeddings:{self.name}",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                message=str(error),
                required=False,
            )
        return ComponentHealth(
            name=f"embeddings:{self.name}",
            status=HealthStatus.HEALTHY,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            required=False,
        )

    async def aclose(self) -> None:
        return None

    @staticmethod
    def validate_texts(texts: Sequence[str]) -> list[str]:
        """Reject input that would silently produce a meaningless vector.

        An empty string embeds to a vector that is *near* everything, which
        shows up later as a retrieval result nobody can explain.
        """
        cleaned = [text.strip() for text in texts]
        if not cleaned:
            raise ValidationError("cannot embed an empty batch")
        if any(not text for text in cleaned):
            raise ValidationError(
                "cannot embed empty text",
                details={"empty_indices": [i for i, t in enumerate(cleaned) if not t]},
            )
        return cleaned


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embeddings from a local Ollama server."""

    name: ClassVar[str] = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        dimensions: int = 768,
        timeout_seconds: float = 60.0,
        client: httpx.AsyncClient | None = None,
        resilience: ResiliencePolicy | None = None,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._client = client or build_async_client(base_url=base_url, timeout=timeout_seconds)
        self._resilience = resilience or ResiliencePolicy(
            timeout_seconds=timeout_seconds,
            retry=RetryPolicy(max_attempts=3, initial_backoff_seconds=0.5),
            breaker=None,
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model(self) -> str:
        return self._model

    async def embed_texts(self, texts: Sequence[str]) -> list[Vector]:
        cleaned = self.validate_texts(texts)

        async def call() -> list[Vector]:
            return await self._embed(cleaned)

        return await self._resilience.execute(call, description="ollama.embed")

    async def _embed(self, texts: list[str]) -> list[Vector]:
        try:
            response = await self._client.post(
                "/api/embed", json={"model": self._model, "input": texts}
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
        except httpx.TimeoutException as error:
            raise AIProviderTimeoutError(
                f"Ollama embedding request timed out: {error}",
                details={"provider": self.name},
            ) from error
        except httpx.HTTPStatusError as error:
            raise AIProviderError(
                f"Ollama embeddings returned HTTP {error.response.status_code}",
                details={"provider": self.name, "model": self._model},
            ) from error
        except httpx.HTTPError as error:
            raise AIProviderError(
                f"Ollama embedding connection error: {error}",
                details={"provider": self.name},
            ) from error

        vectors = payload.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise AIProviderError(
                "Ollama returned an unexpected number of embeddings",
                details={"expected": len(texts), "received": len(vectors or [])},
            )

        # A dimension mismatch against the storage column fails on write with a
        # far less obvious error, so it is caught at the source.
        width = len(vectors[0]) if vectors else 0
        if width != self._dimensions:
            raise AIProviderError(
                "embedding width does not match the configured dimensions",
                details={
                    "configured": self._dimensions,
                    "received": width,
                    "model": self._model,
                },
            )
        return [list(map(float, vector)) for vector in vectors]

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["EmbeddingProvider", "OllamaEmbeddingProvider", "Vector"]
