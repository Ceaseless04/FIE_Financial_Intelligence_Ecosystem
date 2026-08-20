"""Domain invariants: identifiers, scope guards, temporal edges, chunk offsets."""

from __future__ import annotations

from datetime import date

import pytest
from marketmind_fixtures import make_entity, make_source
from pydantic import ValidationError as PydanticValidationError

from fie_schemas.provenance import Provenance
from marketmind.domain.documents import Chunk, Document, DocumentType
from marketmind.domain.entities import Entity, EntityType, Identifier, IdentifierType
from marketmind.domain.relationships import (
    RELATIONSHIP_SCHEMA,
    SYMMETRIC_RELATIONSHIPS,
    Relationship,
    RelationshipType,
)

pytestmark = pytest.mark.unit


class TestIdentifierNormalization:
    def test_cik_zero_padding_is_stripped(self) -> None:
        """The same company must not become two nodes because a feed pads."""
        padded = Identifier(type=IdentifierType.CIK, value="0000320193")
        bare = Identifier(type=IdentifierType.CIK, value="320193")
        assert padded.value == bare.value
        assert padded.key == bare.key

    def test_ticker_is_upper_cased(self) -> None:
        assert Identifier(type=IdentifierType.TICKER, value="acme").value == "ACME"

    def test_domain_is_lower_cased(self) -> None:
        assert Identifier(type=IdentifierType.DOMAIN, value="ACME.COM").value == "acme.com"

    @pytest.mark.parametrize(
        ("identifier_type", "value"),
        [
            (IdentifierType.CIK, "not-a-number"),
            (IdentifierType.TICKER, "1ACME"),
            (IdentifierType.LEI, "TOO-SHORT"),
            (IdentifierType.ISIN, "US123"),
            (IdentifierType.DOMAIN, "no-tld"),
        ],
    )
    def test_malformed_identifiers_are_rejected(
        self, identifier_type: IdentifierType, value: str
    ) -> None:
        with pytest.raises(PydanticValidationError):
            Identifier(type=identifier_type, value=value)


class TestEntityScopeGuard:
    """MarketMind stores knowledge, not judgement — enforced, not documented."""

    @pytest.mark.parametrize(
        "attribute",
        ["rating", "price_target", "fair_value", "investment_score", "portfolio_weight"],
    )
    def test_evaluative_attributes_are_rejected(self, attribute: str) -> None:
        with pytest.raises(PydanticValidationError) as excinfo:
            make_entity("Acme Robotics", attributes={attribute: 42})
        assert "knowledge, not judgements" in str(excinfo.value)

    def test_rejection_is_case_insensitive(self) -> None:
        with pytest.raises(PydanticValidationError):
            make_entity("Acme Robotics", attributes={"Price_Target": 100})

    def test_descriptive_attributes_are_allowed(self) -> None:
        entity = make_entity(
            "Acme Robotics", attributes={"sector": "Industrials", "employee_count": 4200}
        )
        assert entity.attributes["sector"] == "Industrials"

    def test_entity_requires_provenance(self) -> None:
        """An entity nobody can source is one nobody can defend in a report."""
        with pytest.raises(PydanticValidationError):
            Entity(type=EntityType.COMPANY, name="Acme Robotics")  # type: ignore[call-arg]


class TestEntityMerge:
    def test_merge_unions_aliases_identifiers_and_sources(self) -> None:
        left = make_entity(
            "Acme Robotics Corporation",
            identifiers=[(IdentifierType.TICKER, "ACME")],
            source_id="doc-1",
        )
        right = make_entity(
            "Acme Robotics Corp",
            identifiers=[(IdentifierType.CIK, "123456")],
            aliases=["Acme"],
            source_id="doc-2",
        )

        merged = left.merged_with(right)

        assert merged.id == left.id
        assert "Acme Robotics Corp" in merged.aliases
        assert "Acme" in merged.aliases
        assert merged.identifier_keys == {"ticker:ACME", "cik:123456"}
        assert {source.source_id for source in merged.provenance.sources} == {"doc-1", "doc-2"}

    def test_first_observed_attribute_wins(self) -> None:
        """Ingestion order must not silently rewrite an established fact."""
        left = make_entity("Acme", attributes={"sector": "Industrials"})
        right = make_entity("Acme", attributes={"sector": "Technology", "founded": "1994"})

        merged = left.merged_with(right)

        assert merged.attributes["sector"] == "Industrials"
        assert merged.attributes["founded"] == "1994"

    def test_merging_different_types_is_refused(self) -> None:
        company = make_entity("Acme", entity_type=EntityType.COMPANY)
        person = make_entity("Acme", entity_type=EntityType.EXECUTIVE)
        with pytest.raises(ValueError, match="different types"):
            company.merged_with(person)

    def test_merge_does_not_duplicate_an_identifier(self) -> None:
        left = make_entity("Acme", identifiers=[(IdentifierType.TICKER, "ACME")])
        right = make_entity("Acme Corp", identifiers=[(IdentifierType.TICKER, "acme")])
        merged = left.merged_with(right)
        assert len(merged.identifiers) == 1


class TestRelationshipSchema:
    def test_every_relationship_type_declares_its_endpoints(self) -> None:
        """A missing entry would raise KeyError at write time, not review time."""
        assert set(RELATIONSHIP_SCHEMA) == set(RelationshipType)

    def test_nonsensical_endpoints_are_rejected(self) -> None:
        with pytest.raises(PydanticValidationError, match="cannot originate from"):
            Relationship(
                type=RelationshipType.EXECUTIVE_OF,
                source_entity_id="a",
                target_entity_id="b",
                source_entity_type=EntityType.INDUSTRY,
                target_entity_type=EntityType.COMPANY,
                provenance=Provenance.fact(make_source()),
            )

    def test_self_loops_are_rejected(self) -> None:
        with pytest.raises(PydanticValidationError, match="itself"):
            Relationship(
                type=RelationshipType.SUPPLIES,
                source_entity_id="same",
                target_entity_id="same",
                source_entity_type=EntityType.COMPANY,
                target_entity_type=EntityType.COMPANY,
                provenance=Provenance.fact(make_source()),
            )

    def test_validity_window_must_not_run_backwards(self) -> None:
        with pytest.raises(PydanticValidationError, match="valid_to must not precede"):
            Relationship(
                type=RelationshipType.SUPPLIES,
                source_entity_id="a",
                target_entity_id="b",
                source_entity_type=EntityType.COMPANY,
                target_entity_type=EntityType.COMPANY,
                valid_from=date(2025, 1, 1),
                valid_to=date(2024, 1, 1),
                provenance=Provenance.fact(make_source()),
            )

    def test_symmetric_edges_share_a_dedupe_key_in_both_directions(self) -> None:
        """Ingestion order must not produce two edges for one mutual fact."""
        forward = Relationship(
            type=RelationshipType.COMPETES_WITH,
            source_entity_id="ent-a",
            target_entity_id="ent-b",
            source_entity_type=EntityType.COMPANY,
            target_entity_type=EntityType.COMPANY,
            provenance=Provenance.fact(make_source()),
        )
        reverse = forward.model_copy(
            update={"source_entity_id": "ent-b", "target_entity_id": "ent-a"}
        )
        assert forward.dedupe_key == reverse.dedupe_key

    def test_directed_edges_do_not_share_a_dedupe_key(self) -> None:
        supplies = Relationship(
            type=RelationshipType.SUPPLIES,
            source_entity_id="ent-a",
            target_entity_id="ent-b",
            source_entity_type=EntityType.COMPANY,
            target_entity_type=EntityType.COMPANY,
            provenance=Provenance.fact(make_source()),
        )
        reverse = supplies.model_copy(
            update={"source_entity_id": "ent-b", "target_entity_id": "ent-a"}
        )
        assert supplies.dedupe_key != reverse.dedupe_key

    def test_symmetric_set_is_a_subset_of_known_types(self) -> None:
        assert SYMMETRIC_RELATIONSHIPS <= set(RelationshipType)

    def test_open_ended_relationship_is_current(self) -> None:
        relationship = Relationship(
            type=RelationshipType.SUPPLIES,
            source_entity_id="a",
            target_entity_id="b",
            source_entity_type=EntityType.COMPANY,
            target_entity_type=EntityType.COMPANY,
            valid_from=date(2020, 1, 1),
            provenance=Provenance.fact(make_source()),
        )
        assert relationship.is_current
        assert not relationship.model_copy(update={"valid_to": date(2024, 1, 1)}).is_current


class TestDocumentAndChunk:
    def test_content_hash_is_stable_and_content_derived(self) -> None:
        first = Document(type=DocumentType.NEWS_ARTICLE, title="A", content="same body")
        second = Document(type=DocumentType.SEC_FILING, title="B", content="same body")
        assert first.content_hash == second.content_hash
        assert first.id != second.id

    def test_chunk_offsets_must_match_its_text(self) -> None:
        """A mismatched span makes every citation from it point elsewhere."""
        with pytest.raises(PydanticValidationError, match="does not match its character span"):
            Chunk(document_id="doc", index=0, text="hello", start_char=0, end_char=99)

    def test_chunk_resolves_back_to_the_document(self) -> None:
        document = Document(type=DocumentType.NEWS_ARTICLE, title="T", content="alpha beta gamma")
        chunk = Chunk(document_id=document.id, index=0, text="beta", start_char=6, end_char=10)
        assert chunk.resolve_in(document) == "beta"

    def test_chunk_refuses_a_foreign_document(self) -> None:
        document = Document(type=DocumentType.NEWS_ARTICLE, title="T", content="abcdef")
        chunk = Chunk(document_id="other", index=0, text="abc", start_char=0, end_char=3)
        with pytest.raises(ValueError, match="does not belong"):
            chunk.resolve_in(document)

    def test_source_reference_carries_the_span_and_excerpt(self) -> None:
        document = Document(type=DocumentType.SEC_FILING, title="10-K", content="alpha beta gamma")
        chunk = Chunk(
            document_id=document.id,
            index=0,
            text="beta",
            start_char=6,
            end_char=10,
            section="ITEM 1",
        )
        reference = chunk.to_source_reference(document)
        assert reference.source_id == document.id
        assert reference.locator is not None
        assert (reference.locator.start_char, reference.locator.end_char) == (6, 10)
        assert reference.locator.section == "ITEM 1"
        assert reference.excerpt == "beta"
