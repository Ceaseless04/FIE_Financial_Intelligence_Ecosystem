"""Cypher construction: the injection guard and the traversal bounds.

Labels and relationship types are the only things ever interpolated into a query
string, because Cypher cannot parameterize them. These tests exist to keep that
allowlist the sole path.
"""

from __future__ import annotations

import pytest

from fie_common.errors import ValidationError
from marketmind.domain.entities import EntityType
from marketmind.domain.relationships import RelationshipType
from marketmind.graph import queries, schema

pytestmark = pytest.mark.unit


class TestLabelAllowlist:
    @pytest.mark.parametrize("entity_type", list(EntityType))
    def test_every_domain_label_is_accepted(self, entity_type: EntityType) -> None:
        assert queries.validate_label(str(entity_type)) == str(entity_type)

    @pytest.mark.parametrize(
        "label",
        [
            "Company) DETACH DELETE n //",
            "Company`",
            "'; MATCH (n) DETACH DELETE n; //",
            "company",
            "",
            "Person",
        ],
    )
    def test_injection_attempts_are_refused(self, label: str) -> None:
        with pytest.raises(ValidationError, match="unknown node label"):
            queries.validate_label(label)

    @pytest.mark.parametrize("relationship_type", list(RelationshipType))
    def test_every_domain_relationship_type_is_accepted(
        self, relationship_type: RelationshipType
    ) -> None:
        assert queries.validate_relationship_type(str(relationship_type)) == str(relationship_type)

    @pytest.mark.parametrize(
        "relationship_type", ["SUPPLIES]->() DETACH DELETE n //", "supplies", "OWNS"]
    )
    def test_bad_relationship_types_are_refused(self, relationship_type: str) -> None:
        with pytest.raises(ValidationError, match="unknown relationship type"):
            queries.validate_relationship_type(relationship_type)

    def test_query_builders_reject_a_bad_label(self) -> None:
        for builder in (
            queries.upsert_entity_query,
            queries.upsert_entity_query_portable,
            queries.find_entity_by_identifier_query,
            queries.find_entity_by_id_query,
            queries.find_entities_by_canonical_name_query,
        ):
            with pytest.raises(ValidationError):
                builder("Company; DROP")

    def test_relationship_builder_validates_all_three_names(self) -> None:
        with pytest.raises(ValidationError):
            queries.upsert_relationship_query("SUPPLIES", "Company", "Malicious")
        with pytest.raises(ValidationError):
            queries.upsert_relationship_query("HACKS", "Company", "Company")


class TestQueryShape:
    def test_values_are_bound_as_parameters(self) -> None:
        """No caller-supplied value may appear as a literal in a query."""
        query = queries.upsert_entity_query_portable("Company")
        for parameter in ("$id", "$name", "$canonical_name", "$source_ids", "$attributes"):
            assert parameter in query

    def test_upsert_preserves_created_at_on_rewrite(self) -> None:
        query = queries.upsert_entity_query_portable("Company")
        assert "ON CREATE SET" in query
        assert "ON MATCH SET" in query
        assert "n.created_at = $now" in query.replace("\n", " ").replace("        ", " ")

    def test_portable_upsert_avoids_apoc(self) -> None:
        assert "apoc." not in queries.upsert_entity_query_portable("Company")
        assert "apoc." in queries.upsert_entity_query("Company")

    def test_search_returns_the_label(self) -> None:
        """The API needs the label to render a typed result."""
        assert "labels(n)[0] AS label" in queries.search_entities_query(None)
        assert "labels(n)[0] AS label" in queries.search_entities_query("Company")

    @pytest.mark.parametrize("depth", [0, 4, -1, 99])
    def test_traversal_depth_is_bounded(self, depth: int) -> None:
        """An unbounded traversal on a dense financial graph never returns."""
        with pytest.raises(ValidationError, match="between 1 and 3"):
            queries.neighbourhood_query(depth, None)

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_allowed_depths_produce_a_bounded_pattern(self, depth: int) -> None:
        query = queries.neighbourhood_query(depth, None)
        assert f"*1..{depth}" in query
        assert "$limit" in query

    def test_relationship_filter_is_validated(self) -> None:
        with pytest.raises(ValidationError):
            queries.neighbourhood_query(1, ["SUPPLIES", "NONSENSE"])
        assert ":SUPPLIES|COMPETES_WITH" in queries.neighbourhood_query(
            1, ["SUPPLIES", "COMPETES_WITH"]
        )

    @pytest.mark.parametrize("direction", ["both", "incoming", "outgoing"])
    def test_known_directions_are_accepted(self, direction: str) -> None:
        assert "$entity_id" in queries.entity_relationships_query(direction)

    def test_unknown_direction_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="unknown direction"):
            queries.entity_relationships_query("sideways")

    @pytest.mark.parametrize("depth", [0, 7])
    def test_shortest_path_depth_is_bounded(self, depth: int) -> None:
        with pytest.raises(ValidationError, match="between 1 and 6"):
            queries.shortest_path_query(depth)


class TestSchemaStatements:
    def test_every_label_gets_a_uniqueness_constraint(self) -> None:
        statements = " ".join(schema.all_schema_statements())
        for entity_type in EntityType:
            assert f":{entity_type.value}" in statements

    def test_statements_are_idempotent(self) -> None:
        """Schema application runs on every startup; it must not fail twice."""
        for statement in schema.all_schema_statements():
            assert "IF NOT EXISTS" in statement.upper()
