"""HTTP contract: authentication, authorization, validation, and error shape.

The application under test is the real one — real routing, real dependency
graph, real error handlers. Only the stores behind it are stubbed, because what
is being asserted is the contract a client sees, and a client cannot tell
whether Neo4j was involved in a 403.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from fie_auth import AuthSettings, Principal, Role, TokenType, create_token
from fie_common.config import CoreSettings, Environment
from fie_common.errors import NotFoundError
from fie_schemas.health import ComponentHealth, HealthStatus
from marketmind.api.app import CORRELATION_HEADER, create_app
from marketmind.api.dependencies import get_session
from marketmind.config import MarketMindSettings
from marketmind.domain.entities import EntityType
from marketmind.graph.repository import GraphEdge, GraphNode, Neighbourhood
from marketmind.retrieval.graphrag import GraphRAGAnswer, RetrievedContext
from marketmind.storage.repository import ChunkHit

pytestmark = pytest.mark.api

AUTH = AuthSettings()

ACME_NODE: dict[str, Any] = {
    "id": "ent-acme",
    "label": "Company",
    "name": "Acme Robotics Corporation",
    "canonical_name": "acme robotics",
    "aliases": ["Acme"],
    "identifier_keys": ["ticker:ACME", "cik:123456"],
    "source_ids": ["doc-1"],
    "sector": "Industrials",
}


@dataclass
class StubGraph:
    nodes: dict[str, dict[str, Any]] = field(default_factory=lambda: {"ent-acme": dict(ACME_NODE)})
    edges: list[GraphEdge] = field(
        default_factory=lambda: [
            GraphEdge(
                type="SUPPLIES",
                relationship_id="rel-1",
                other_id="ent-northwind",
                other_label="Company",
                other_name="Northwind Components Ltd",
                is_outgoing=False,
                source_ids=["doc-1"],
            )
        ]
    )
    path: dict[str, Any] | None = None

    async def find_by_id(
        self, entity_id: str, *, entity_type: EntityType | None = None
    ) -> dict[str, Any] | None:
        return self.nodes.get(entity_id)

    async def search(
        self, term: str, *, entity_type: EntityType | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        return [node for node in self.nodes.values() if term.lower() in str(node["name"]).lower()][
            :limit
        ]

    async def relationships_of(
        self, entity_id: str, *, direction: str = "both", limit: int = 100
    ) -> list[GraphEdge]:
        return list(self.edges) if entity_id in self.nodes else []

    async def neighbourhood(self, entity_id: str, **_kwargs: Any) -> Neighbourhood:
        return Neighbourhood(
            seed_entity_id=entity_id,
            nodes=[
                GraphNode(
                    id="ent-northwind",
                    label="Company",
                    name="Northwind Components Ltd",
                    distance=1,
                    source_ids=["doc-1"],
                )
            ],
        )

    async def shortest_path(
        self, source_id: str, target_id: str, *, max_depth: int = 4
    ) -> dict[str, Any] | None:
        return self.path

    async def stats(self) -> dict[str, Any]:
        return {
            "nodes_by_label": {"Company": 2},
            "relationships_by_type": {"SUPPLIES": 1},
            "total_nodes": 2,
            "total_relationships": 1,
        }


@dataclass
class StubGraphRAG:
    answer_result: GraphRAGAnswer | None = None
    context_result: RetrievedContext | None = None
    error: Exception | None = None

    async def retrieve(self, question: str, **_kwargs: Any) -> RetrievedContext:
        if self.error is not None:
            raise self.error
        return self.context_result or RetrievedContext(
            question=question,
            chunks=[
                ChunkHit(
                    chunk_id="chk-1",
                    document_id="doc-1",
                    text="Northwind supplies Acme with precision bearings.",
                    start_char=0,
                    end_char=47,
                    distance=0.2,
                )
            ],
            graph_facts=["Northwind Components supplies Acme Robotics"],
            entity_ids=["ent-acme"],
        )

    async def answer(self, question: str, **_kwargs: Any) -> GraphRAGAnswer:
        if self.error is not None:
            raise self.error
        assert self.answer_result is not None
        return self.answer_result


@dataclass
class StubPipeline:
    report: Any = None

    async def ingest(self, document: Any, **_kwargs: Any) -> Any:
        from marketmind.ingestion.pipeline import IngestionReport

        return self.report or IngestionReport(
            document_id=document.id,
            chunk_count=3,
            embedded_chunk_count=3,
            entities_created=2,
            relationships_created=1,
            embeddings_written=True,
        )


@dataclass
class StubContainer:
    """Presents the ServiceContainer surface the routers actually use."""

    settings: MarketMindSettings = field(default_factory=MarketMindSettings)
    core: CoreSettings = field(default_factory=CoreSettings)
    auth: AuthSettings = field(default_factory=AuthSettings)
    graph: StubGraph = field(default_factory=StubGraph)
    rag: StubGraphRAG = field(default_factory=StubGraphRAG)
    ingestion: StubPipeline = field(default_factory=StubPipeline)
    components: list[ComponentHealth] = field(
        default_factory=lambda: [
            ComponentHealth(name="postgres", status=HealthStatus.HEALTHY),
            ComponentHealth(name="neo4j", status=HealthStatus.HEALTHY),
            ComponentHealth(name="embeddings:ollama", status=HealthStatus.HEALTHY, required=False),
        ]
    )

    def graphrag(self, _session: Any) -> StubGraphRAG:
        return self.rag

    def pipeline(self, _session: Any) -> StubPipeline:
        return self.ingestion

    async def health(self) -> list[ComponentHealth]:
        return list(self.components)

    async def aclose(self) -> None:
        return None


def token_for(*roles: Role, permissions: list[str] | None = None) -> str:
    principal = Principal(subject="tester", roles=list(roles), permissions=permissions or [])
    return create_token(principal, AUTH, token_type=TokenType.ACCESS)


def auth_header(*roles: Role, permissions: list[str] | None = None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token_for(*roles, permissions=permissions)}"}


@pytest.fixture
def container() -> StubContainer:
    return StubContainer()


@pytest.fixture
async def client(container: StubContainer) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(container=container, configure_observability=False)  # type: ignore[arg-type]
    # The stub stores need no session; overriding the dependency keeps the rest
    # of the real dependency graph intact.
    app.dependency_overrides[get_session] = lambda: None

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://marketmind.test"
        ) as http_client:
            yield http_client


class TestHealth:
    async def test_liveness_needs_no_authentication(self, client: httpx.AsyncClient) -> None:
        """A liveness probe cannot be expected to carry a token."""
        response = await client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}

    async def test_health_reports_components(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/health")
        body = response.json()

        assert response.status_code == 200
        assert body["status"] == "healthy"
        assert {component["name"] for component in body["components"]} == {
            "postgres",
            "neo4j",
            "embeddings:ollama",
        }

    async def test_readiness_returns_503_when_a_required_component_is_down(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        container.components = [
            ComponentHealth(name="postgres", status=HealthStatus.UNHEALTHY, required=True)
        ]

        response = await client.get("/health/ready")

        assert response.status_code == 503
        assert response.json()["status"] == "unhealthy"

    async def test_readiness_stays_200_when_only_an_optional_component_is_down(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        """Losing the model must not take entity lookups out of rotation."""
        container.components = [
            ComponentHealth(name="postgres", status=HealthStatus.HEALTHY),
            ComponentHealth(name="ai:claude", status=HealthStatus.UNHEALTHY, required=False),
        ]

        response = await client.get("/health/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "degraded"


class TestAuthentication:
    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/entities/ent-acme",
            "/api/v1/entities/search?q=acme",
            "/api/v1/graph/stats",
            "/api/v1/graph/visualization/ent-acme",
        ],
    )
    async def test_protected_reads_require_a_token(
        self, client: httpx.AsyncClient, path: str
    ) -> None:
        response = await client.get(path)
        assert response.status_code == 401
        assert response.json()["code"] == "authentication_error"

    async def test_ingestion_requires_a_token(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/documents",
            json={"type": "news_article", "title": "T", "content": "Body text."},
        )
        assert response.status_code == 401

    async def test_a_malformed_scheme_is_rejected(self, client: httpx.AsyncClient) -> None:
        response = await client.get(
            "/api/v1/graph/stats", headers={"Authorization": "Basic abc123"}
        )
        assert response.status_code == 401
        assert response.json()["details"]["reason"] == "invalid_scheme"

    async def test_a_garbage_token_is_rejected(self, client: httpx.AsyncClient) -> None:
        response = await client.get(
            "/api/v1/graph/stats", headers={"Authorization": "Bearer not.a.jwt"}
        )
        assert response.status_code == 401

    async def test_a_refresh_token_cannot_be_used_as_an_access_token(
        self, client: httpx.AsyncClient
    ) -> None:
        """Token type is pinned; a refresh token must not open the API."""
        refresh = create_token(
            Principal(subject="tester", roles=[Role.ANALYST]),
            AUTH,
            token_type=TokenType.REFRESH,
        )

        response = await client.get(
            "/api/v1/graph/stats", headers={"Authorization": f"Bearer {refresh}"}
        )

        assert response.status_code == 401


class TestAuthorization:
    async def test_a_viewer_may_read(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/graph/stats", headers=auth_header(Role.VIEWER))
        assert response.status_code == 200

    async def test_a_viewer_may_not_ingest(self, client: httpx.AsyncClient) -> None:
        """Read access to the graph does not imply the right to change it."""
        response = await client.post(
            "/api/v1/documents",
            json={"type": "news_article", "title": "T", "content": "Body text."},
            headers=auth_header(Role.VIEWER),
        )
        assert response.status_code == 403
        assert response.json()["code"] == "authorization_error"

    async def test_an_operator_may_ingest(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/documents",
            json={"type": "news_article", "title": "T", "content": "Body text."},
            headers=auth_header(Role.OPERATOR),
        )
        assert response.status_code == 201

    async def test_an_admin_may_do_both(self, client: httpx.AsyncClient) -> None:
        assert (
            await client.get("/api/v1/graph/stats", headers=auth_header(Role.ADMIN))
        ).status_code == 200
        assert (
            await client.post(
                "/api/v1/documents",
                json={"type": "news_article", "title": "T", "content": "Body."},
                headers=auth_header(Role.ADMIN),
            )
        ).status_code == 201

    async def test_a_principal_with_no_roles_is_denied(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/graph/stats", headers=auth_header())
        assert response.status_code == 403


class TestEntityEndpoints:
    async def test_fetching_an_entity_returns_its_identifiers_and_sources(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.get("/api/v1/entities/ent-acme", headers=auth_header(Role.ANALYST))
        body = response.json()

        assert response.status_code == 200
        assert body["id"] == "ent-acme"
        assert body["type"] == "Company"
        assert body["identifiers"] == {"ticker": "ACME", "cik": "123456"}
        assert body["source_document_ids"] == ["doc-1"]
        assert body["attributes"]["sector"] == "Industrials"

    async def test_an_unknown_entity_is_404(self, client: httpx.AsyncClient) -> None:
        response = await client.get(
            "/api/v1/entities/ent-missing", headers=auth_header(Role.ANALYST)
        )
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    async def test_search_returns_typed_results(self, client: httpx.AsyncClient) -> None:
        response = await client.get(
            "/api/v1/entities/search?q=acme", headers=auth_header(Role.ANALYST)
        )
        body = response.json()

        assert response.status_code == 200
        assert body[0]["name"] == "Acme Robotics Corporation"
        assert body[0]["type"] == "Company"

    async def test_search_rejects_an_empty_query(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/entities/search?q=", headers=auth_header(Role.ANALYST))
        assert response.status_code == 422

    async def test_search_limit_is_bounded(self, client: httpx.AsyncClient) -> None:
        response = await client.get(
            "/api/v1/entities/search?q=acme&limit=5000", headers=auth_header(Role.ANALYST)
        )
        assert response.status_code == 422

    async def test_relationships_report_direction_and_sources(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.get(
            "/api/v1/entities/ent-acme/relationships", headers=auth_header(Role.ANALYST)
        )
        body = response.json()

        assert response.status_code == 200
        assert body[0]["direction"] == "incoming"
        assert body[0]["other_entity_name"] == "Northwind Components Ltd"
        assert body[0]["source_document_ids"] == ["doc-1"]

    async def test_an_unknown_direction_is_rejected(self, client: httpx.AsyncClient) -> None:
        response = await client.get(
            "/api/v1/entities/ent-acme/relationships?direction=sideways",
            headers=auth_header(Role.ANALYST),
        )
        assert response.status_code == 422


class TestGraphEndpoints:
    async def test_visualization_returns_nodes_and_directed_edges(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.get(
            "/api/v1/graph/visualization/ent-acme", headers=auth_header(Role.ANALYST)
        )
        body = response.json()

        assert response.status_code == 200
        assert {node["id"] for node in body["nodes"]} == {"ent-acme", "ent-northwind"}
        seed = next(node for node in body["nodes"] if node["id"] == "ent-acme")
        assert seed["distance"] == 0
        # The traversal reported an incoming SUPPLIES edge, so the renderer must
        # receive it pointing at the seed rather than away from it.
        assert body["edges"][0] == {
            "id": "rel-1",
            "type": "SUPPLIES",
            "source": "ent-northwind",
            "target": "ent-acme",
            "source_document_ids": ["doc-1"],
        }

    async def test_visualization_depth_is_capped(self, client: httpx.AsyncClient) -> None:
        response = await client.get(
            "/api/v1/graph/visualization/ent-acme?depth=9",
            headers=auth_header(Role.ANALYST),
        )
        assert response.status_code == 422

    async def test_visualization_of_an_unknown_entity_is_404(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.get(
            "/api/v1/graph/visualization/ent-missing", headers=auth_header(Role.ANALYST)
        )
        assert response.status_code == 404

    async def test_path_returns_the_connecting_chain(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        container.graph.path = {
            "length": 2,
            "nodes": [
                {"id": "ent-northwind", "name": "Northwind Components", "label": "Company"},
                {"id": "ent-acme", "name": "Acme Robotics", "label": "Company"},
                {"id": "ent-zenith", "name": "Zenith Automation", "label": "Company"},
            ],
            "relationships": [{"type": "SUPPLIES"}, {"type": "COMPETES_WITH"}],
        }

        response = await client.get(
            "/api/v1/graph/path?source=ent-northwind&target=ent-zenith",
            headers=auth_header(Role.ANALYST),
        )
        body = response.json()

        assert response.status_code == 200
        assert body["length"] == 2
        assert body["nodes"][0]["name"] == "Northwind Components"
        assert body["relationship_types"] == ["SUPPLIES", "COMPETES_WITH"]

    async def test_no_path_is_404_not_an_empty_result(self, client: httpx.AsyncClient) -> None:
        """An empty path object would read as "they are connected by nothing"."""
        response = await client.get(
            "/api/v1/graph/path?source=ent-acme&target=ent-unrelated",
            headers=auth_header(Role.ANALYST),
        )
        assert response.status_code == 404

    async def test_stats_are_returned(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/graph/stats", headers=auth_header(Role.ANALYST))
        assert response.json()["total_nodes"] == 2


class TestRetrievalEndpoints:
    def _answer(self, **overrides: Any) -> GraphRAGAnswer:
        from fie_schemas.provenance import Provenance, SourceReference, SourceType

        defaults: dict[str, Any] = {
            "question": "Who supplies Acme?",
            "answer": "Northwind Components supplies Acme [doc-1].",
            "provenance": Provenance.generated(
                "claude-opus-5",
                SourceReference(
                    source_id="doc-1",
                    source_type=SourceType.SEC_FILING,
                    title="Acme 10-K",
                    uri="https://example.invalid/acme",
                ),
            ),
            "cited_source_ids": ["doc-1"],
            "inferences": [],
            "entity_ids": ["ent-acme"],
            "sufficient_context": True,
            "grounded": True,
        }
        return GraphRAGAnswer(**{**defaults, **overrides})

    async def test_an_answer_returns_resolvable_citations(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        container.rag.answer_result = self._answer()

        response = await client.post(
            "/api/v1/retrieval/answer",
            json={"question": "Who supplies Acme?"},
            headers=auth_header(Role.ANALYST),
        )
        body = response.json()

        assert response.status_code == 200
        assert body["grounded"] is True
        assert body["citations"] == [
            {
                "document_id": "doc-1",
                "title": "Acme 10-K",
                "uri": "https://example.invalid/acme",
            }
        ]
        assert body["provenance"]["kind"] == "generated"
        assert body["provenance"]["model"] == "claude-opus-5"

    async def test_an_ungrounded_answer_says_so(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        """A client must be able to distinguish a fully-sourced answer."""
        container.rag.answer_result = self._answer(grounded=False, cited_source_ids=[])

        response = await client.post(
            "/api/v1/retrieval/answer",
            json={"question": "Who supplies Acme?"},
            headers=auth_header(Role.ANALYST),
        )

        assert response.json()["grounded"] is False

    async def test_no_context_is_404(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        container.rag.error = NotFoundError("no relevant context found for the question")

        response = await client.post(
            "/api/v1/retrieval/answer",
            json={"question": "Something never indexed"},
            headers=auth_header(Role.ANALYST),
        )

        assert response.status_code == 404

    async def test_context_endpoint_exposes_the_retrieved_spans(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.post(
            "/api/v1/retrieval/context",
            json={"question": "Who supplies Acme?"},
            headers=auth_header(Role.ANALYST),
        )
        body = response.json()

        assert response.status_code == 200
        assert body["passages"][0]["document_id"] == "doc-1"
        assert body["passages"][0]["start_char"] == 0
        assert 0.0 <= body["passages"][0]["similarity"] <= 1.0
        assert body["graph_facts"]

    async def test_a_too_short_question_is_rejected(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/retrieval/answer",
            json={"question": "a"},
            headers=auth_header(Role.ANALYST),
        )
        assert response.status_code == 422

    async def test_unknown_fields_are_rejected(self, client: httpx.AsyncClient) -> None:
        """An unexpected field crossing a service boundary is a contract break."""
        response = await client.post(
            "/api/v1/retrieval/answer",
            json={"question": "Who supplies Acme?", "temperature": 0.9},
            headers=auth_header(Role.ANALYST),
        )
        assert response.status_code == 422


class TestIngestionEndpoint:
    async def test_ingestion_reports_what_it_did(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/documents",
            json={
                "type": "sec_filing",
                "title": "Acme Robotics 10-K",
                "content": "Acme Robotics Corporation (NASDAQ: ACME) filed its annual report.",
            },
            headers=auth_header(Role.OPERATOR),
        )
        body = response.json()

        assert response.status_code == 201
        assert body["chunk_count"] == 3
        assert body["entities_created"] == 2
        assert body["is_retrievable"] is True
        assert body["rejected"] == []

    async def test_a_duplicate_is_reported_not_hidden(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        from marketmind.ingestion.pipeline import IngestionReport

        container.ingestion.report = IngestionReport(document_id="doc-1", duplicate=True)

        response = await client.post(
            "/api/v1/documents",
            json={"type": "news_article", "title": "T", "content": "Body text."},
            headers=auth_header(Role.OPERATOR),
        )

        assert response.status_code == 201
        assert response.json()["duplicate"] is True

    async def test_rejected_extractions_are_surfaced(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        """A run that dropped half its edges must not look like a clean success."""
        from marketmind.ingestion.pipeline import IngestionReport

        container.ingestion.report = IngestionReport(
            document_id="doc-1",
            chunk_count=2,
            embedded_chunk_count=2,
            rejected=["relationship SUPPLIES is not valid between Industry and Executive"],
            embeddings_written=True,
        )

        response = await client.post(
            "/api/v1/documents",
            json={"type": "news_article", "title": "T", "content": "Body text."},
            headers=auth_header(Role.OPERATOR),
        )

        assert response.json()["rejected"]

    @pytest.mark.parametrize(
        "payload",
        [
            {"type": "not_a_type", "title": "T", "content": "Body."},
            {"type": "news_article", "title": "", "content": "Body."},
            {"type": "news_article", "title": "T", "content": ""},
            {"type": "news_article", "title": "T"},
        ],
    )
    async def test_malformed_payloads_are_rejected(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> None:
        response = await client.post(
            "/api/v1/documents", json=payload, headers=auth_header(Role.OPERATOR)
        )
        assert response.status_code == 422
        assert response.json()["code"] == "validation_error"


class TestErrorEnvelopeAndCorrelation:
    async def test_errors_use_the_shared_envelope(self, client: httpx.AsyncClient) -> None:
        response = await client.get(
            "/api/v1/entities/ent-missing", headers=auth_header(Role.ANALYST)
        )
        body = response.json()

        assert set(body) >= {"code", "message", "retryable", "details"}
        assert body["retryable"] is False

    async def test_an_inbound_correlation_id_is_echoed(self, client: httpx.AsyncClient) -> None:
        """One id must follow a request that began in another service."""
        response = await client.get("/health/live", headers={CORRELATION_HEADER: "corr-from-atlas"})
        assert response.headers[CORRELATION_HEADER] == "corr-from-atlas"

    async def test_a_correlation_id_is_minted_when_absent(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/health/live")
        assert response.headers[CORRELATION_HEADER].startswith("corr")

    async def test_an_unexpected_error_does_not_leak_internals(
        self, container: StubContainer
    ) -> None:
        container.rag.error = RuntimeError("psycopg: connection to 10.0.0.5:5432 failed")
        app = create_app(container=container, configure_observability=False)  # type: ignore[arg-type]
        app.dependency_overrides[get_session] = lambda: None

        # Starlette generates the 500 body from the handler and then re-raises so
        # the server can log it. Under uvicorn that is what puts the traceback in
        # the logs; here it has to be suppressed to inspect what the client got.
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://marketmind.test"
            ) as isolated:
                response = await isolated.post(
                    "/api/v1/retrieval/context",
                    json={"question": "Who supplies Acme?"},
                    headers=auth_header(Role.ANALYST),
                )

        body = response.json()
        assert response.status_code == 500
        assert body["code"] == "internal_error"
        assert "10.0.0.5" not in body["message"]
        assert "psycopg" not in body["message"]
        assert body["correlation_id"]


class TestProductionPosture:
    async def test_interactive_docs_are_disabled_in_production(self) -> None:
        """The schema is published deliberately or not at all."""
        production = CoreSettings(environment=Environment.PRODUCTION, debug=False)
        app = create_app(
            container=StubContainer(),  # type: ignore[arg-type]
            core=production,
            configure_observability=False,
        )
        assert app.docs_url is None
        assert app.openapi_url is None

    async def test_docs_are_available_in_development(self) -> None:
        app = create_app(
            container=StubContainer(),  # type: ignore[arg-type]
            core=CoreSettings(environment=Environment.DEVELOPMENT),
            configure_observability=False,
        )
        assert app.docs_url == "/docs"
