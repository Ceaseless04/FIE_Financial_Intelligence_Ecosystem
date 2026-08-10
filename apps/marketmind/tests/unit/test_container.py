"""The composition root.

Wiring is easy to get wrong in ways no other test notices: a health check that
marks an optional dependency required takes a serving instance out of rotation,
and a shutdown that stops at the first failure leaks connections. Both are
asserted here against stub collaborators, so the container's own logic is under
test rather than the clients it holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from fie_auth import AuthSettings
from fie_common.config import CoreSettings, Environment
from fie_common.errors import ConfigurationError
from fie_schemas.health import ComponentHealth, HealthStatus
from marketmind.api.dependencies import (
    PERMISSION_READ_ENTITY,
    PERMISSION_READ_GRAPH,
    PERMISSION_READ_RETRIEVAL,
    PERMISSION_WRITE_INGESTION,
    ServiceContainer,
    readiness_status,
)
from marketmind.config import MarketMindSettings
from marketmind.graph.repository import GraphRepository
from marketmind.ingestion.extraction import ExtractionService
from marketmind.resolution.resolver import EntityResolver

pytestmark = pytest.mark.unit


@dataclass
class Recorder:
    """Stub client that records whether it was closed."""

    name: str
    closed: bool = False
    status: HealthStatus = HealthStatus.HEALTHY
    required: bool = True
    fail_close: bool = False

    async def health_check(self) -> ComponentHealth:
        return ComponentHealth(name=self.name, status=self.status, required=self.required)

    async def aclose(self) -> None:
        if self.fail_close:
            raise RuntimeError(f"{self.name} refused to close")
        self.closed = True


@dataclass
class RecordingRegistry:
    components: list[ComponentHealth] = field(default_factory=list)
    closed: bool = False

    async def health_check(self) -> list[ComponentHealth]:
        return list(self.components)

    async def aclose(self) -> None:
        self.closed = True


@dataclass
class RecordingRouter:
    registry: RecordingRegistry = field(default_factory=RecordingRegistry)


def build_container(**overrides: Any) -> tuple[ServiceContainer, dict[str, Any]]:
    parts: dict[str, Any] = {
        "database": Recorder("postgres"),
        "neo4j": Recorder("neo4j"),
        "embeddings": Recorder("embeddings:ollama", required=False),
        "router": RecordingRouter(),
    }
    parts.update(overrides)
    container = ServiceContainer(
        settings=MarketMindSettings(),
        core=CoreSettings(),
        auth=AuthSettings(),
        database=parts["database"],
        neo4j=parts["neo4j"],
        embeddings=parts["embeddings"],
        router=parts["router"],
        graph=GraphRepository(client=None),  # type: ignore[arg-type]
        extraction=ExtractionService(parts["router"]),  # type: ignore[arg-type]
        resolver=EntityResolver(),
    )
    return container, parts


class TestHealthComposition:
    async def test_every_dependency_is_probed(self) -> None:
        container, _ = build_container()

        components = await container.health()

        assert {component.name for component in components} == {
            "postgres",
            "neo4j",
            "embeddings:ollama",
        }

    async def test_ai_providers_are_reported_as_optional(self) -> None:
        """MarketMind still serves lookups and traversals without a model."""
        registry = RecordingRegistry(
            components=[
                ComponentHealth(name="ai:claude", status=HealthStatus.UNHEALTHY, required=True)
            ]
        )
        container, _ = build_container(router=RecordingRouter(registry=registry))

        components = await container.health()

        provider = next(c for c in components if c.name == "ai:claude")
        assert provider.required is False

    async def test_readiness_is_unhealthy_only_on_a_required_failure(self) -> None:
        assert (
            readiness_status(
                [ComponentHealth(name="postgres", status=HealthStatus.UNHEALTHY, required=True)]
            )
            is HealthStatus.UNHEALTHY
        )
        assert (
            readiness_status(
                [ComponentHealth(name="ai", status=HealthStatus.UNHEALTHY, required=False)]
            )
            is HealthStatus.DEGRADED
        )
        assert (
            readiness_status([ComponentHealth(name="postgres", status=HealthStatus.HEALTHY)])
            is HealthStatus.HEALTHY
        )


class TestShutdown:
    async def test_every_resource_is_released(self) -> None:
        container, parts = build_container()

        await container.aclose()

        assert parts["database"].closed
        assert parts["neo4j"].closed
        assert parts["embeddings"].closed
        assert parts["router"].registry.closed

    async def test_one_failing_close_does_not_strand_the_others(self) -> None:
        """A client that throws on shutdown must not leak the remaining ones."""
        container, parts = build_container(
            neo4j=Recorder("neo4j", fail_close=True),
        )

        await container.aclose()

        assert parts["database"].closed
        assert parts["embeddings"].closed
        assert parts["router"].registry.closed


class TestFactories:
    def test_the_pipeline_is_configured_from_settings(self) -> None:
        container, _ = build_container()
        container.settings = MarketMindSettings(  # type: ignore[misc]
            chunk_max_chars=900,
            chunk_overlap_chars=90,
            embedding_batch_size=7,
            extraction_max_chunks=11,
        )

        pipeline = container.pipeline(session=None)  # type: ignore[arg-type]

        assert pipeline.chunking.max_chars == 900
        assert pipeline.embedding_batch_size == 7
        assert pipeline.max_extraction_chunks == 11

    def test_graphrag_is_configured_from_settings(self) -> None:
        container, _ = build_container()
        container.settings = MarketMindSettings(retrieval_top_k_chunks=3)  # type: ignore[misc]

        service = container.graphrag(session=None)  # type: ignore[arg-type]

        assert service.config.top_k_chunks == 3


class TestProductionGuardrails:
    def test_build_refuses_an_unconfigured_default_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failing at startup beats failing at the first request."""
        monkeypatch.setenv("FIE_AI_DEFAULT_PROVIDER", "claude")
        monkeypatch.setenv("FIE_AI_ENABLED_PROVIDERS", '["claude"]')
        monkeypatch.delenv("FIE_CLAUDE_API_KEY", raising=False)

        with pytest.raises(ConfigurationError):
            ServiceContainer.build(core=CoreSettings(environment=Environment.DEVELOPMENT))

    def test_build_refuses_a_mismatched_embedding_width(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MARKETMIND_EMBEDDING_DIMENSIONS", "1536")

        with pytest.raises(ConfigurationError, match="pgvector column width"):
            ServiceContainer.build(core=CoreSettings(environment=Environment.DEVELOPMENT))


class TestPermissionConstants:
    def test_permissions_are_namespaced_to_marketmind(self) -> None:
        for permission in (
            PERMISSION_READ_ENTITY,
            PERMISSION_READ_GRAPH,
            PERMISSION_READ_RETRIEVAL,
            PERMISSION_WRITE_INGESTION,
        ):
            assert permission.startswith("marketmind:")

    def test_reads_and_writes_are_distinct_grants(self) -> None:
        """Read access to the graph must not imply the right to change it."""
        reads = {PERMISSION_READ_ENTITY, PERMISSION_READ_GRAPH, PERMISSION_READ_RETRIEVAL}
        assert PERMISSION_WRITE_INGESTION not in reads
        assert all(permission.endswith(":read") for permission in reads)
