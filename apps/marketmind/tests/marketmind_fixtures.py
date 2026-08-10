"""Builders shared across the MarketMind suites.

Named for the app rather than something generic. Test helper modules are
imported by bare module name — the app's ``tests`` directory is on ``sys.path``
— and ``sys.modules`` is process-wide, so two apps that both offered a module
called ``conftest`` would silently resolve to whichever loaded first. That
failure only appears once a second app exists, and then it appears as every test
in one app erroring at setup.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from fie_ai.contracts import CompletionResponse, StopReason, TokenUsage
from fie_ai.registry import AIRouter, ProviderRegistry
from fie_schemas.provenance import Provenance, SourceReference, SourceType
from fie_testing import FakeAIProvider
from marketmind.domain.entities import Entity, EntityType, Identifier, IdentifierType


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
