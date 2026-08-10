"""Event contracts and configuration guardrails.

The event payloads are a published contract: Atlas, Sentinel, and Venture read
them. These tests pin the shape so a rename shows up here rather than as a
missing key in another service.
"""

from __future__ import annotations

import pytest

from fie_common.config import Environment
from fie_common.errors import ConfigurationError
from fie_events.bus import stream_for
from fie_events.schemas import DomainEvent
from marketmind import events
from marketmind.config import MarketMindSettings
from marketmind.storage.models import EMBEDDING_DIMENSIONS

pytestmark = pytest.mark.unit


class TestEventContract:
    def test_every_published_type_is_namespaced_to_marketmind(self) -> None:
        for event_type in events.PUBLISHED_EVENT_TYPES:
            assert event_type.startswith("marketmind.")

    def test_every_published_type_routes_to_one_stream(self) -> None:
        """Consumers subscribe per-product; a stray namespace would be missed."""
        streams = {stream_for(event_type) for event_type in events.PUBLISHED_EVENT_TYPES}
        assert streams == {"fie.events.marketmind"}

    def test_event_types_parse_as_product_entity_action(self) -> None:
        for event_type in events.PUBLISHED_EVENT_TYPES:
            event = DomainEvent(event_type=event_type, source_app=events.SOURCE_APP)
            assert event.product == "marketmind"
            assert len(event_type.split(".")) == 3

    def test_merge_event_names_both_ids(self) -> None:
        """A consumer holding the retired id must learn what replaced it."""
        payload = events.EntityMergedPayload(
            surviving_entity_id="ent-keep",
            merged_entity_id="ent-drop",
            entity_type="Company",
            name="Acme Robotics",
            match_reason="identifier_match",
            confidence=1.0,
            evidence="shared identifier(s): ['cik']",
        )
        event = events.build_event(events.ENTITY_MERGED, payload)

        assert event.payload["surviving_entity_id"] == "ent-keep"
        assert event.payload["merged_entity_id"] == "ent-drop"
        assert event.payload["match_reason"] == "identifier_match"

    def test_confidence_is_bounded(self) -> None:
        with pytest.raises(ValueError):
            events.EntityMergedPayload(
                surviving_entity_id="a",
                merged_entity_id="b",
                entity_type="Company",
                name="Acme",
                match_reason="fuzzy_name_match",
                confidence=1.4,
            )

    def test_payloads_serialize_to_json_primitives(self) -> None:
        """Events cross a Redis stream as strings; a stray object breaks the wire."""
        payload = events.DocumentIngestedPayload(
            document_id="doc-1",
            document_type="sec_filing",
            title="10-K",
            content_hash="abc123",
            chunk_count=12,
            embedded_chunk_count=12,
            entity_count=4,
            relationship_count=3,
        )
        event = events.build_event(events.DOCUMENT_INGESTED, payload, correlation_id="corr-1")
        wire = event.to_wire()

        assert all(isinstance(value, str) for value in wire.values())
        assert DomainEvent.from_wire(wire).payload["chunk_count"] == 12

    def test_correlation_id_is_carried(self) -> None:
        event = events.build_event(
            events.GRAPH_UPDATED,
            events.GraphUpdatedPayload(
                document_id="doc-1",
                entities_created=1,
                entities_merged=0,
                relationships_created=2,
                rejected_count=0,
            ),
            correlation_id="corr-abc",
        )
        assert event.correlation_id == "corr-abc"
        assert event.source_app == "marketmind"

    def test_counts_cannot_be_negative(self) -> None:
        with pytest.raises(ValueError):
            events.GraphUpdatedPayload(
                document_id="doc-1",
                entities_created=-1,
                entities_merged=0,
                relationships_created=0,
                rejected_count=0,
            )


class TestSettings:
    def test_defaults_are_development_safe(self) -> None:
        settings = MarketMindSettings()
        settings.validate_for(Environment.DEVELOPMENT)
        assert settings.embedding_dimensions == EMBEDDING_DIMENSIONS

    def test_embedding_width_must_match_the_column(self) -> None:
        """pgvector fixes the width in a migration; a mismatch fails every insert."""
        settings = MarketMindSettings(embedding_dimensions=1536)
        with pytest.raises(ConfigurationError, match="pgvector column width"):
            settings.validate_for(Environment.DEVELOPMENT)

    def test_overlap_must_be_smaller_than_the_chunk(self) -> None:
        settings = MarketMindSettings(chunk_max_chars=500, chunk_overlap_chars=500)
        with pytest.raises(ConfigurationError, match="smaller than the chunk size"):
            settings.validate_for(Environment.DEVELOPMENT)

    def test_localhost_embeddings_are_refused_in_production(self) -> None:
        settings = MarketMindSettings(embedding_base_url="http://localhost:11434")
        with pytest.raises(ConfigurationError, match="localhost in production"):
            settings.validate_for(Environment.PRODUCTION)

    def test_production_accepts_a_real_embedding_host(self) -> None:
        settings = MarketMindSettings(embedding_base_url="https://embeddings.internal:8000")
        settings.validate_for(Environment.PRODUCTION)

    def test_component_configs_are_derived_from_settings(self) -> None:
        settings = MarketMindSettings(
            chunk_max_chars=800,
            chunk_overlap_chars=80,
            chunk_min_chars=40,
            resolution_fuzzy_threshold=95.0,
            retrieval_top_k_chunks=9,
            retrieval_graph_depth=2,
        )

        assert settings.chunking().max_chars == 800
        assert settings.chunking().overlap_chars == 80
        assert settings.resolution().fuzzy_threshold == 95.0
        assert settings.graphrag().top_k_chunks == 9
        assert settings.graphrag().graph_depth == 2

    def test_traversal_depth_cannot_exceed_the_query_bound(self) -> None:
        """Settings must not permit a depth the Cypher builder will reject."""
        with pytest.raises(ValueError):
            MarketMindSettings(retrieval_graph_depth=4)

    def test_settings_are_immutable_once_loaded(self) -> None:
        settings = MarketMindSettings()
        with pytest.raises(ValueError):
            settings.embedding_model = "something-else"  # type: ignore[misc]
