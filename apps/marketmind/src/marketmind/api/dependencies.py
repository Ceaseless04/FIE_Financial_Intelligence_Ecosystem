"""Composition root and FastAPI dependencies.

Every collaborator is constructed once here and hung off the application state.
Routers ask for what they need through dependencies and never import a client
directly, which is what lets the API tests run the real routing logic against
fake providers instead of standing up four containers to check a status code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from fie_ai.embeddings import EmbeddingProvider, OllamaEmbeddingProvider
from fie_ai.factory import build_router
from fie_ai.registry import AIRouter
from fie_auth import AuthSettings, Principal, authenticate, parse_bearer_header
from fie_common.config import CoreSettings, Environment
from fie_database.config import Neo4jSettings, PostgresSettings, RedisSettings
from fie_database.neo4j_client import Neo4jClient
from fie_database.postgres import PostgresDatabase
from fie_events.bus import EventBus
from fie_observability.logging import get_logger
from fie_schemas.health import ComponentHealth, HealthStatus
from marketmind.config import MarketMindSettings
from marketmind.graph.repository import GraphRepository
from marketmind.ingestion.extraction import ExtractionService
from marketmind.ingestion.pipeline import IngestionPipeline
from marketmind.resolution.resolver import EntityResolver
from marketmind.retrieval.graphrag import GraphRAGService
from marketmind.storage.repository import DocumentRepository

logger = get_logger(__name__)


@dataclass
class ServiceContainer:
    """Everything the API needs, wired once at startup."""

    settings: MarketMindSettings
    core: CoreSettings
    auth: AuthSettings
    database: PostgresDatabase
    neo4j: Neo4jClient
    embeddings: EmbeddingProvider
    router: AIRouter
    graph: GraphRepository
    extraction: ExtractionService
    resolver: EntityResolver
    bus: EventBus | None = None
    _owned: list[object] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        *,
        settings: MarketMindSettings | None = None,
        core: CoreSettings | None = None,
        bus: EventBus | None = None,
    ) -> ServiceContainer:
        """Construct the production wiring from environment configuration."""
        core = core or CoreSettings()
        settings = settings or MarketMindSettings()
        settings.validate_for(core.environment)

        auth = AuthSettings()
        auth.validate_for(core.environment)

        postgres = PostgresSettings()
        postgres.validate_for(core.environment)
        neo4j_settings = Neo4jSettings()
        neo4j_settings.validate_for(core.environment)

        embeddings = OllamaEmbeddingProvider(
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
        ai_router = build_router(environment=core.environment)
        neo4j_client = Neo4jClient(neo4j_settings)

        return cls(
            settings=settings,
            core=core,
            auth=auth,
            database=PostgresDatabase(postgres),
            neo4j=neo4j_client,
            embeddings=embeddings,
            router=ai_router,
            graph=GraphRepository(neo4j_client, use_apoc=settings.graph_use_apoc),
            extraction=ExtractionService(
                ai_router,
                max_chunks_per_call=settings.extraction_chunks_per_call,
                max_tokens=settings.extraction_max_tokens,
            ),
            resolver=EntityResolver(settings.resolution()),
            bus=bus,
        )

    def documents(self, session: AsyncSession) -> DocumentRepository:
        return DocumentRepository(session)

    def pipeline(self, session: AsyncSession) -> IngestionPipeline:
        return IngestionPipeline(
            documents=self.documents(session),
            graph=self.graph,
            embeddings=self.embeddings,
            extraction=self.extraction,
            resolver=self.resolver,
            bus=self.bus,
            chunking=self.settings.chunking(),
            embedding_batch_size=self.settings.embedding_batch_size,
            max_extraction_chunks=self.settings.extraction_max_chunks,
        )

    def graphrag(self, session: AsyncSession) -> GraphRAGService:
        return GraphRAGService(
            documents=self.documents(session),
            graph=self.graph,
            embeddings=self.embeddings,
            router=self.router,
            config=self.settings.graphrag(),
        )

    async def health(self) -> list[ComponentHealth]:
        """Check every dependency.

        The AI provider and the embedding server are marked non-required: a
        MarketMind that cannot synthesize an answer can still serve entity
        lookups and traversals, and reporting it as fully down would take the
        service out of rotation for capability it still has.
        """
        components = [
            await self.database.health_check(),
            await self.neo4j.health_check(),
        ]
        embedding_health = await self.embeddings.health_check()
        components.append(embedding_health)

        for provider_health in await self.router.registry.health_check():
            components.append(
                provider_health.model_copy(update={"required": False})
                if provider_health.required
                else provider_health
            )
        return components

    async def aclose(self) -> None:
        """Release every resource, continuing past individual failures."""
        for closer in (
            self.embeddings.aclose,
            self.neo4j.aclose,
            self.database.aclose,
            self.router.registry.aclose,
        ):
            try:
                await closer()
            except Exception as error:  # noqa: BLE001 — shutdown must complete
                logger.warning("shutdown_close_failed", error=str(error))
        if self.bus is not None:
            try:
                await self.bus.aclose()
            except Exception as error:  # noqa: BLE001
                logger.warning("shutdown_close_failed", component="bus", error=str(error))


def get_container(request: Request) -> ServiceContainer:
    """The container built during application startup."""
    container: ServiceContainer = request.app.state.container
    return container


ContainerDep = Annotated[ServiceContainer, Depends(get_container)]


async def get_session(container: ContainerDep) -> AsyncIterator[AsyncSession]:
    """A transaction-scoped session, committed when the handler returns."""
    async with container.database.session() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_principal(request: Request, container: ContainerDep) -> Principal:
    """Authenticate the caller from the ``Authorization`` header.

    Raises:
        AuthenticationError: if the header is missing, malformed, or the token
            fails verification. Translated to 401 by the app's error handler.
    """
    token = parse_bearer_header(request.headers.get("authorization"))
    return authenticate(token, container.auth)


PrincipalDep = Annotated[Principal, Depends(get_principal)]


def require_permission(permission: str) -> Callable[[Principal], Principal]:
    """Dependency factory enforcing a single permission.

    Authorization is declared per route rather than checked inside handlers, so
    an unprotected endpoint is visible in the routing table instead of hiding in
    a function body.
    """

    def dependency(principal: PrincipalDep) -> Principal:
        principal.require_permission(permission)
        return principal

    return dependency


#: Permissions this API enforces. Named constants so the router declarations and
#: the tests that assert on them cannot drift apart.
PERMISSION_READ_ENTITY = "marketmind:entity:read"
PERMISSION_READ_GRAPH = "marketmind:graph:read"
PERMISSION_READ_RETRIEVAL = "marketmind:retrieval:read"
PERMISSION_WRITE_INGESTION = "marketmind:ingestion:write"


def readiness_status(components: list[ComponentHealth]) -> HealthStatus:
    """Roll component health into a readiness verdict."""
    if any(c.status is HealthStatus.UNHEALTHY and c.required for c in components):
        return HealthStatus.UNHEALTHY
    if any(c.status is not HealthStatus.HEALTHY for c in components):
        return HealthStatus.DEGRADED
    return HealthStatus.HEALTHY


__all__ = [
    "PERMISSION_READ_ENTITY",
    "PERMISSION_READ_GRAPH",
    "PERMISSION_READ_RETRIEVAL",
    "PERMISSION_WRITE_INGESTION",
    "ContainerDep",
    "Environment",
    "PrincipalDep",
    "RedisSettings",
    "ServiceContainer",
    "SessionDep",
    "get_container",
    "get_principal",
    "get_session",
    "readiness_status",
    "require_permission",
]
