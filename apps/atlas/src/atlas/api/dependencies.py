"""Composition root and FastAPI dependencies.

Every collaborator is constructed once here and hung off the application state.
Routers ask for what they need through dependencies and never import a client
directly, which is what lets the API tests run the real routing, validation, and
authorization logic against fake providers instead of standing up four
containers to check a status code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from atlas.clients.marketmind import MarketMindClient
from atlas.config import AtlasSettings
from atlas.filings.extraction import StatementExtractor
from atlas.filings.pipeline import FilingPipeline
from atlas.research.pipeline import ResearchPipeline
from atlas.research.reports import ReportService
from atlas.storage.repository import FilingRepository, ReportRepository, StatementRepository
from fie_ai.factory import build_router
from fie_ai.registry import AIRouter
from fie_auth import AuthSettings, Principal, authenticate, parse_bearer_header
from fie_common.config import CoreSettings, Environment
from fie_database.config import PostgresSettings, RedisSettings
from fie_database.postgres import PostgresDatabase
from fie_events.bus import EventBus
from fie_observability.logging import get_logger
from fie_schemas.health import ComponentHealth, HealthStatus

logger = get_logger(__name__)


@dataclass
class ServiceContainer:
    """Everything the API needs, wired once at startup."""

    settings: AtlasSettings
    core: CoreSettings
    auth: AuthSettings
    database: PostgresDatabase
    router: AIRouter
    extractor: StatementExtractor
    reports: ReportService
    marketmind: MarketMindClient | None = None
    bus: EventBus | None = None
    _owned: list[object] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        *,
        settings: AtlasSettings | None = None,
        core: CoreSettings | None = None,
        bus: EventBus | None = None,
    ) -> ServiceContainer:
        """Construct the production wiring from environment configuration."""
        core = core or CoreSettings()
        settings = settings or AtlasSettings()
        settings.validate_for(core.environment)

        auth = AuthSettings()
        auth.validate_for(core.environment)

        postgres = PostgresSettings()
        postgres.validate_for(core.environment)

        ai_router = build_router(environment=core.environment)

        marketmind = (
            MarketMindClient(
                base_url=settings.marketmind_base_url,
                token=settings.marketmind_token,
                timeout_seconds=settings.marketmind_timeout_seconds,
            )
            if settings.marketmind_enabled
            else None
        )

        return cls(
            settings=settings,
            core=core,
            auth=auth,
            database=PostgresDatabase(postgres),
            router=ai_router,
            extractor=StatementExtractor(
                ai_router,
                max_tokens=settings.extraction_max_tokens,
                max_content_chars=settings.extraction_max_content_chars,
            ),
            reports=ReportService(
                ai_router,
                max_tokens=settings.report_max_tokens,
                max_regenerations=settings.report_max_regenerations,
            ),
            marketmind=marketmind,
            bus=bus,
        )

    # -- per-request collaborators -------------------------------------------

    def filings(self, session: AsyncSession) -> FilingRepository:
        return FilingRepository(session)

    def statements(self, session: AsyncSession) -> StatementRepository:
        return StatementRepository(session)

    def report_store(self, session: AsyncSession) -> ReportRepository:
        return ReportRepository(session)

    def ingestion(self, session: AsyncSession) -> FilingPipeline:
        return FilingPipeline(
            filings=self.filings(session),
            statements=self.statements(session),
            extractor=self.extractor,
            bus=self.bus,
        )

    def research(self, session: AsyncSession) -> ResearchPipeline:
        return ResearchPipeline(
            statements=self.statements(session),
            reports=self.report_store(session),
            service=self.reports,
            marketmind=self.marketmind,
            bus=self.bus,
        )

    async def health(self) -> list[ComponentHealth]:
        """Check every dependency.

        Postgres is the only required one. Atlas without an AI provider can
        still serve stored filings, stored statements, and every deterministic
        ratio and valuation — which is most of the product. Reporting it as
        fully down would take the service out of rotation for capability it
        demonstrably still has.
        """
        components = [await self.database.health_check()]

        for provider_health in await self.router.registry.health_check():
            components.append(
                provider_health.model_copy(update={"required": False})
                if provider_health.required
                else provider_health
            )

        if self.marketmind is not None:
            components.append(await self.marketmind.health_check())
        return components

    async def aclose(self) -> None:
        """Release every resource, continuing past individual failures."""
        closers = [self.database.aclose, self.router.registry.aclose]
        if self.marketmind is not None:
            closers.append(self.marketmind.aclose)
        for closer in closers:
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
PERMISSION_WRITE_FILING = "atlas:filing:write"
PERMISSION_READ_FILING = "atlas:filing:read"
PERMISSION_READ_ANALYSIS = "atlas:analysis:read"
#: Valuation is separated from analysis deliberately. A ratio is a fact about a
#: filing; a DCF is an estimate that reads like one, and an organisation may
#: reasonably let more people see the first than the second.
PERMISSION_READ_VALUATION = "atlas:valuation:read"
PERMISSION_WRITE_RESEARCH = "atlas:research:write"
PERMISSION_READ_RESEARCH = "atlas:research:read"


def readiness_status(components: list[ComponentHealth]) -> HealthStatus:
    """Roll component health into a readiness verdict."""
    if any(c.status is HealthStatus.UNHEALTHY and c.required for c in components):
        return HealthStatus.UNHEALTHY
    if any(c.status is not HealthStatus.HEALTHY for c in components):
        return HealthStatus.DEGRADED
    return HealthStatus.HEALTHY


__all__ = [
    "PERMISSION_READ_ANALYSIS",
    "PERMISSION_READ_FILING",
    "PERMISSION_READ_RESEARCH",
    "PERMISSION_READ_VALUATION",
    "PERMISSION_WRITE_FILING",
    "PERMISSION_WRITE_RESEARCH",
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
