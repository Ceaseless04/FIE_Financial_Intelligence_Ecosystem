"""Composition root and FastAPI dependencies.

Every collaborator is constructed once here and hung off the application state.
Routers ask for what they need through dependencies and never import a client
directly, which is what lets the API tests run the real routing, validation, and
authorization logic against fakes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from cfo_ai.analysis.compare import ComparisonSettings
from cfo_ai.config import CFOSettings
from cfo_ai.pipeline import PlanningPipeline, ReportingPipeline
from cfo_ai.research.commentary import CommentaryService
from cfo_ai.storage.repository import ChartRepository, CommentaryRepository, PlanRepository
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

    settings: CFOSettings
    core: CoreSettings
    auth: AuthSettings
    database: PostgresDatabase
    router: AIRouter
    commentary_service: CommentaryService
    bus: EventBus | None = None
    _owned: list[object] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        *,
        settings: CFOSettings | None = None,
        core: CoreSettings | None = None,
        bus: EventBus | None = None,
    ) -> ServiceContainer:
        """Construct the production wiring from environment configuration."""
        core = core or CoreSettings()
        settings = settings or CFOSettings()
        settings.validate_for(core.environment)

        auth = AuthSettings()
        auth.validate_for(core.environment)

        postgres = PostgresSettings()
        postgres.validate_for(core.environment)

        ai_router = build_router(environment=core.environment)

        return cls(
            settings=settings,
            core=core,
            auth=auth,
            database=PostgresDatabase(postgres),
            router=ai_router,
            commentary_service=CommentaryService(
                ai_router,
                max_tokens=settings.commentary_max_tokens,
                max_regenerations=settings.commentary_max_regenerations,
            ),
            bus=bus,
        )

    # -- per-request collaborators -------------------------------------------

    def charts(self, session: AsyncSession) -> ChartRepository:
        return ChartRepository(session)

    def plans(self, session: AsyncSession) -> PlanRepository:
        return PlanRepository(session)

    def commentaries(self, session: AsyncSession) -> CommentaryRepository:
        return CommentaryRepository(session)

    def comparison_settings(self) -> ComparisonSettings:
        return ComparisonSettings(
            material_percent=self.settings.material_percent,
            material_amount=self.settings.material_amount,
        )

    def planning(self, session: AsyncSession) -> PlanningPipeline:
        return PlanningPipeline(
            charts=self.charts(session), plans=self.plans(session), bus=self.bus
        )

    def reporting(self, session: AsyncSession) -> ReportingPipeline:
        return ReportingPipeline(
            charts=self.charts(session),
            plans=self.plans(session),
            commentaries=self.commentaries(session),
            service=self.commentary_service,
            settings=self.comparison_settings(),
            bus=self.bus,
        )

    async def health(self) -> list[ComponentHealth]:
        """Check every dependency.

        Postgres is the only required one. CFO.ai without an AI provider still
        stores plans and computes every variance, roll-up, forecast, and
        scenario — which is most of the product, and all of the part that has to
        be right. Only the commentary is lost.
        """
        components = [await self.database.health_check()]
        for provider_health in await self.router.registry.health_check():
            components.append(
                provider_health.model_copy(update={"required": False})
                if provider_health.required
                else provider_health
            )
        return components

    async def aclose(self) -> None:
        """Release every resource, continuing past individual failures."""
        for closer in (self.database.aclose, self.router.registry.aclose):
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


#: Permissions this API enforces. A budget is among the most sensitive documents
#: a company holds, so writing one is separated from reading it, and reading the
#: plan is separated from reading the variance against it — the people who may
#: see how a department performed are not always the people who set its budget.
PERMISSION_WRITE_PLAN = "cfo:plan:write"
PERMISSION_READ_PLAN = "cfo:plan:read"
PERMISSION_READ_VARIANCE = "cfo:variance:read"
PERMISSION_WRITE_COMMENTARY = "cfo:commentary:write"
PERMISSION_READ_COMMENTARY = "cfo:commentary:read"


def readiness_status(components: list[ComponentHealth]) -> HealthStatus:
    """Roll component health into a readiness verdict."""
    if any(c.status is HealthStatus.UNHEALTHY and c.required for c in components):
        return HealthStatus.UNHEALTHY
    if any(c.status is not HealthStatus.HEALTHY for c in components):
        return HealthStatus.DEGRADED
    return HealthStatus.HEALTHY


__all__ = [
    "PERMISSION_READ_COMMENTARY",
    "PERMISSION_READ_PLAN",
    "PERMISSION_READ_VARIANCE",
    "PERMISSION_WRITE_COMMENTARY",
    "PERMISSION_WRITE_PLAN",
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
