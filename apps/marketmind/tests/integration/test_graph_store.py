"""Graph writes and traversals against a real Neo4j.

The properties here cannot be checked against a fake: idempotent MERGE
semantics, list union on re-ingestion, variable-length path bounds, and
shortestPath are all database behaviour. A fake that returns what the repository
asked for would pass these tests against a broken query.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from conftest import make_entity
from fie_database.config import Neo4jSettings
from fie_database.neo4j_client import Neo4jClient
from fie_schemas.provenance import Provenance, SourceReference, SourceType
from fie_testing import neo4j_endpoint, require_service
from marketmind.domain.entities import EntityType, IdentifierType
from marketmind.domain.relationships import Relationship, RelationshipType
from marketmind.graph.repository import GraphRepository, entity_from_node

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


@pytest.fixture(scope="module", autouse=True)
def _requires_neo4j() -> None:
    require_service(neo4j_endpoint())


@pytest.fixture
async def graph() -> AsyncIterator[GraphRepository]:
    client = Neo4jClient(Neo4jSettings())
    repository = GraphRepository(client)
    await repository.delete_all()
    await repository.apply_schema()
    try:
        yield repository
    finally:
        await repository.delete_all()
        await client.aclose()


def company(name: str, *, cik: str | None = None, source_id: str = "doc-1") -> object:
    identifiers = [(IdentifierType.CIK, cik)] if cik else []
    entity = make_entity(name, identifiers=identifiers, source_id=source_id)  # type: ignore[arg-type]
    return entity.model_copy(update={"canonical_name": name.lower()})


def supplies(source: object, target: object, *, source_id: str = "doc-1") -> Relationship:
    return Relationship(
        type=RelationshipType.SUPPLIES,
        source_entity_id=source.id,  # type: ignore[attr-defined]
        target_entity_id=target.id,  # type: ignore[attr-defined]
        source_entity_type=EntityType.COMPANY,
        target_entity_type=EntityType.COMPANY,
        provenance=Provenance.fact(
            SourceReference(source_id=source_id, source_type=SourceType.SEC_FILING)
        ),
    )


class TestSchema:
    async def test_schema_application_is_idempotent(self, graph: GraphRepository) -> None:
        """Startup runs it every time; the second run must be a no-op."""
        first = await graph.apply_schema()
        second = await graph.apply_schema()
        assert first > 0
        assert second == first


class TestIdempotentWrites:
    async def test_re_ingesting_the_same_entity_creates_one_node(
        self, graph: GraphRepository
    ) -> None:
        entity = company("Acme Robotics", cik="123456")

        await graph.upsert_entity(entity)  # type: ignore[arg-type]
        await graph.upsert_entity(entity)  # type: ignore[arg-type]

        stats = await graph.stats()
        assert stats["total_nodes"] == 1

    async def test_re_ingestion_unions_sources_rather_than_replacing(
        self, graph: GraphRepository
    ) -> None:
        """A second filing must add evidence, not erase the first one's."""
        first = company("Acme Robotics", cik="123456", source_id="doc-1")
        second = first.model_copy(  # type: ignore[attr-defined]
            update={
                "provenance": Provenance.fact(
                    SourceReference(source_id="doc-2", source_type=SourceType.SEC_FILING)
                )
            }
        )

        await graph.upsert_entity(first)  # type: ignore[arg-type]
        await graph.upsert_entity(second)

        node = await graph.find_by_id(first.id)  # type: ignore[attr-defined]
        assert node is not None
        assert set(node["source_ids"]) == {"doc-1", "doc-2"}

    async def test_re_ingesting_a_relationship_creates_one_edge(
        self, graph: GraphRepository
    ) -> None:
        acme, northwind = company("Acme Robotics"), company("Northwind Components")
        await graph.upsert_entity(acme)  # type: ignore[arg-type]
        await graph.upsert_entity(northwind)  # type: ignore[arg-type]
        edge = supplies(northwind, acme)

        await graph.upsert_relationship(edge)
        await graph.upsert_relationship(edge)

        stats = await graph.stats()
        assert stats["total_relationships"] == 1

    async def test_symmetric_edges_collapse_regardless_of_direction(
        self, graph: GraphRepository
    ) -> None:
        acme, zenith = company("Acme Robotics"), company("Zenith Automation")
        await graph.upsert_entity(acme)  # type: ignore[arg-type]
        await graph.upsert_entity(zenith)  # type: ignore[arg-type]

        forward = Relationship(
            type=RelationshipType.COMPETES_WITH,
            source_entity_id=acme.id,  # type: ignore[attr-defined]
            target_entity_id=zenith.id,  # type: ignore[attr-defined]
            source_entity_type=EntityType.COMPANY,
            target_entity_type=EntityType.COMPANY,
            provenance=Provenance.fact(
                SourceReference(source_id="doc-1", source_type=SourceType.SEC_FILING)
            ),
        )
        reverse = forward.model_copy(
            update={
                "source_entity_id": zenith.id,  # type: ignore[attr-defined]
                "target_entity_id": acme.id,  # type: ignore[attr-defined]
            }
        )

        await graph.upsert_relationship(forward)
        await graph.upsert_relationship(reverse)

        stats = await graph.stats()
        assert stats["total_relationships"] == 1

    async def test_a_relationship_with_a_missing_endpoint_is_skipped(
        self, graph: GraphRepository
    ) -> None:
        """A dangling edge is a data problem, not a reason to lose the batch."""
        acme = company("Acme Robotics")
        ghost = company("Ghost Corp")
        await graph.upsert_entity(acme)  # type: ignore[arg-type]

        written = await graph.upsert_relationships([supplies(ghost, acme)])

        assert written == []


class TestLookups:
    async def test_find_by_identifier(self, graph: GraphRepository) -> None:
        entity = company("Acme Robotics", cik="123456")
        await graph.upsert_entity(entity)  # type: ignore[arg-type]

        node = await graph.find_by_identifier(EntityType.COMPANY, "cik:123456")

        assert node is not None
        assert node["name"] == "Acme Robotics"

    async def test_find_by_id_without_a_type_hint(self, graph: GraphRepository) -> None:
        entity = company("Acme Robotics")
        await graph.upsert_entity(entity)  # type: ignore[arg-type]

        node = await graph.find_by_id(entity.id)  # type: ignore[attr-defined]

        assert node is not None
        assert node["label"] == "Company"

    async def test_find_by_id_returns_none_when_absent(self, graph: GraphRepository) -> None:
        assert await graph.find_by_id("ent_missing") is None

    async def test_search_returns_the_label(self, graph: GraphRepository) -> None:
        await graph.upsert_entity(company("Acme Robotics"))  # type: ignore[arg-type]

        results = await graph.search("acme")

        assert results
        assert results[0]["label"] == "Company"

    async def test_search_can_be_restricted_by_type(self, graph: GraphRepository) -> None:
        await graph.upsert_entity(company("Acme Robotics"))  # type: ignore[arg-type]
        executive = make_entity("Acme Person", entity_type=EntityType.EXECUTIVE)
        await graph.upsert_entity(executive.model_copy(update={"canonical_name": "acme person"}))

        companies = await graph.search("acme", entity_type=EntityType.COMPANY)

        assert {node["label"] for node in companies} == {"Company"}

    async def test_a_stored_node_rehydrates_into_a_domain_entity(
        self, graph: GraphRepository
    ) -> None:
        original = company("Acme Robotics", cik="123456")
        await graph.upsert_entity(original)  # type: ignore[arg-type]

        node = await graph.find_by_id(original.id)  # type: ignore[attr-defined]
        assert node is not None
        rebuilt = entity_from_node(node, EntityType.COMPANY)

        assert rebuilt.id == original.id  # type: ignore[attr-defined]
        assert rebuilt.identifier_of(IdentifierType.CIK) == "123456"
        assert rebuilt.resolved
        assert rebuilt.provenance.sources


class TestTraversal:
    @pytest.fixture
    async def supply_chain(self, graph: GraphRepository) -> dict[str, object]:
        """Northwind -> Acme -> Zenith, a three-node chain."""
        northwind = company("Northwind Components")
        acme = company("Acme Robotics")
        zenith = company("Zenith Automation")
        for entity in (northwind, acme, zenith):
            await graph.upsert_entity(entity)  # type: ignore[arg-type]
        await graph.upsert_relationship(supplies(northwind, acme))
        await graph.upsert_relationship(supplies(acme, zenith))
        return {"northwind": northwind, "acme": acme, "zenith": zenith}

    async def test_depth_one_reaches_direct_neighbours_only(
        self, graph: GraphRepository, supply_chain: dict[str, object]
    ) -> None:
        result = await graph.neighbourhood(
            supply_chain["northwind"].id,
            depth=1,  # type: ignore[attr-defined]
        )
        assert {node.name for node in result.nodes} == {"Acme Robotics"}

    async def test_depth_two_reaches_the_far_end_of_the_chain(
        self, graph: GraphRepository, supply_chain: dict[str, object]
    ) -> None:
        result = await graph.neighbourhood(
            supply_chain["northwind"].id,
            depth=2,  # type: ignore[attr-defined]
        )
        assert {node.name for node in result.nodes} == {"Acme Robotics", "Zenith Automation"}
        distances = {node.name: node.distance for node in result.nodes}
        assert distances["Acme Robotics"] < distances["Zenith Automation"]

    async def test_traversal_can_be_restricted_to_edge_types(
        self, graph: GraphRepository, supply_chain: dict[str, object]
    ) -> None:
        result = await graph.neighbourhood(
            supply_chain["acme"].id,  # type: ignore[attr-defined]
            depth=1,
            relationship_types=[RelationshipType.COMPETES_WITH],
        )
        assert result.nodes == []

    async def test_relationship_direction_is_reported(
        self, graph: GraphRepository, supply_chain: dict[str, object]
    ) -> None:
        edges = await graph.relationships_of(
            supply_chain["acme"].id  # type: ignore[attr-defined]
        )
        by_name = {edge.other_name: edge for edge in edges}
        assert by_name["Northwind Components"].is_outgoing is False
        assert by_name["Zenith Automation"].is_outgoing is True

    async def test_direction_filter_narrows_the_result(
        self, graph: GraphRepository, supply_chain: dict[str, object]
    ) -> None:
        outgoing = await graph.relationships_of(
            supply_chain["acme"].id,
            direction="outgoing",  # type: ignore[attr-defined]
        )
        assert {edge.other_name for edge in outgoing} == {"Zenith Automation"}

    async def test_shortest_path_explains_an_indirect_connection(
        self, graph: GraphRepository, supply_chain: dict[str, object]
    ) -> None:
        """This is the query Sentinel turns into an exposure explanation."""
        path = await graph.shortest_path(
            supply_chain["northwind"].id,  # type: ignore[attr-defined]
            supply_chain["zenith"].id,  # type: ignore[attr-defined]
        )

        assert path is not None
        assert path["length"] == 2
        assert [node["name"] for node in path["nodes"]] == [
            "Northwind Components",
            "Acme Robotics",
            "Zenith Automation",
        ]

    async def test_no_path_returns_none(
        self, graph: GraphRepository, supply_chain: dict[str, object]
    ) -> None:
        isolated = company("Isolated Holdings")
        await graph.upsert_entity(isolated)  # type: ignore[arg-type]

        path = await graph.shortest_path(
            supply_chain["acme"].id,
            isolated.id,  # type: ignore[attr-defined]
        )

        assert path is None

    async def test_stats_count_by_label_and_type(
        self, graph: GraphRepository, supply_chain: dict[str, object]
    ) -> None:
        stats = await graph.stats()
        assert stats["nodes_by_label"]["Company"] == 3
        assert stats["relationships_by_type"]["SUPPLIES"] == 2


class TestWriteGuard:
    async def test_the_read_path_refuses_a_write_query(self, graph: GraphRepository) -> None:
        """Defence in depth: a read helper must not be able to mutate."""
        from fie_common.errors import ValidationError

        with pytest.raises(ValidationError):
            await graph.client.execute_read("MATCH (n) DETACH DELETE n")
