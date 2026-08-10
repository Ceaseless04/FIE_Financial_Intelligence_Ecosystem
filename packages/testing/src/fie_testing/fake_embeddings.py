"""Deterministic embedding provider for tests.

Produces real unit vectors with meaningful geometry — texts sharing tokens land
closer together — from a hash of their tokens. That makes similarity search
assertable without a model server, and identical across runs and machines.

It is not semantic: "revenue" and "sales" are unrelated here. Tests that depend
on semantic closeness belong in the integration suite against a real model.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import ClassVar

from fie_ai.embeddings import EmbeddingProvider, Vector

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Hash-based embeddings: stable, fast, and geometrically sane."""

    name: ClassVar[str] = "deterministic"

    def __init__(self, *, dimensions: int = 64, model: str = "deterministic-v1") -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self._dimensions = dimensions
        self._model = model
        self.call_count = 0
        self.embedded_texts: list[str] = []

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model(self) -> str:
        return self._model

    async def embed_texts(self, texts: Sequence[str]) -> list[Vector]:
        cleaned = self.validate_texts(texts)
        self.call_count += 1
        self.embedded_texts.extend(cleaned)
        return [self._embed_one(text) for text in cleaned]

    def _embed_one(self, text: str) -> Vector:
        """Sum a per-token basis vector, then normalize.

        Shared tokens therefore raise cosine similarity, which is the property
        retrieval tests actually rely on.
        """
        vector = [0.0] * self._dimensions
        tokens = _TOKEN_PATTERN.findall(text.lower()) or [text.lower()]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(self._dimensions):
                byte = digest[index % len(digest)]
                # Map bytes onto [-1, 1] so unrelated tokens can cancel out
                # instead of every vector pointing into the same orthant.
                vector[index] += (byte / 127.5) - 1.0

        magnitude = math.sqrt(sum(component * component for component in vector))
        if magnitude == 0.0:  # pragma: no cover — only for a pathological hash
            vector[0] = 1.0
            return vector
        return [component / magnitude for component in vector]


def cosine_similarity(left: Vector, right: Vector) -> float:
    """Cosine similarity between two vectors, for assertions in tests."""
    if len(left) != len(right):
        raise ValueError("vectors must have the same width")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_magnitude = math.sqrt(sum(a * a for a in left))
    right_magnitude = math.sqrt(sum(b * b for b in right))
    if left_magnitude == 0.0 or right_magnitude == 0.0:
        return 0.0
    return dot / (left_magnitude * right_magnitude)


__all__ = ["DeterministicEmbeddingProvider", "cosine_similarity"]
