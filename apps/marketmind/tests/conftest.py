"""Shared fixtures for the MarketMind suites."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from fie_ai.contracts import CompletionResponse, StopReason, TokenUsage
from fie_ai.registry import AIRouter, ProviderRegistry
from fie_schemas.provenance import Provenance, SourceReference, SourceType
from fie_testing import DeterministicEmbeddingProvider, FakeAIProvider
from marketmind.domain.entities import Entity, EntityType, Identifier, IdentifierType

# The fixtures package is imported by path rather than installed, so tests can
# say `from fixtures.documents import acme_10k` under importlib import mode.
sys.path.insert(0, str(Path(__file__).parent))


def make_source(source_id: str = "doc-1") -> SourceReference:
    return SourceReference(source_id=source_id, source_type=SourceType.SEC_FILING)


def make_entity(
    name: str,
    *,
    entity_type: EntityType = EntityType.COMPANY,
    identifiers: Sequence[tuple[IdentifierType, str]] = (),
    aliases: Sequence[str] = (),
    source_id: str = "doc-1",
    attributes: dict[str, Any] | None = None,
) -> Entity:
    """Build an entity with valid provenance.

    Every entity in the domain requires a source, so tests that only care about
    names would otherwise repeat the same provenance boilerplate everywhere.
    """
    return Entity(
        type=entity_type,
        name=name,
        aliases=list(aliases),
        identifiers=[
            Identifier(type=identifier_type, value=value) for identifier_type, value in identifiers
        ],
        attributes=attributes or {},
        provenance=Provenance.fact(make_source(source_id)),
    )


def make_json_response(
    payload: dict[str, Any], *, model: str = "fake-model-1"
) -> CompletionResponse:
    """A completion whose text is the JSON a structured request expects."""
    return CompletionResponse(
        text=json.dumps(payload),
        model=model,
        provider="fake",
        stop_reason=StopReason.END_TURN,
        usage=TokenUsage(input_tokens=10, output_tokens=20),
    )


def make_router(responses: Sequence[CompletionResponse | Exception]) -> AIRouter:
    """A router backed by one scripted fake provider."""
    registry = ProviderRegistry()
    registry.register(FakeAIProvider(list(responses)), default=True)
    return AIRouter(registry, primary="fake")


@pytest.fixture
def embeddings() -> DeterministicEmbeddingProvider:
    """Deterministic embeddings with real vector geometry."""
    return DeterministicEmbeddingProvider(dimensions=64)


@pytest.fixture
def source() -> SourceReference:
    return make_source()
