"""Health, readiness, and liveness endpoints.

Three separate checks because orchestrators use them differently: liveness
answers "should this process be restarted", readiness answers "should traffic be
routed here", and health is the human-facing detail. Conflating them produces
restart loops during a dependency outage the process would have survived.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from cfo_ai.api.dependencies import ContainerDep
from fie_schemas.health import HealthReport, HealthStatus

router = APIRouter(tags=["health"])


@router.get("/health/live", summary="Liveness probe")
async def live() -> dict[str, str]:
    """Whether the process is running.

    Deliberately checks nothing else: a dependency being down is not a reason
    to kill and restart a healthy process.
    """
    return {"status": "alive"}


@router.get("/health", response_model=HealthReport, summary="Full health report")
async def health(container: ContainerDep) -> HealthReport:
    """Detailed dependency health."""
    return HealthReport.from_components(
        service=container.settings.service_name,
        version=container.core.service_version,
        environment=str(container.core.environment),
        components=await container.health(),
    )


@router.get("/health/ready", response_model=HealthReport, summary="Readiness probe")
async def ready(container: ContainerDep, response: Response) -> HealthReport:
    """Whether this instance should receive traffic.

    Returns 503 only when Postgres is unreachable. Degraded — the AI provider
    is down — still returns 200, because every stored plan and every
    deterministic variance, roll-up, forecast, and scenario continues to work.
    That is most of the product and all of the part that has to be right; only
    the commentary is lost.
    """
    report = HealthReport.from_components(
        service=container.settings.service_name,
        version=container.core.service_version,
        environment=str(container.core.environment),
        components=await container.health(),
    )
    if report.status is HealthStatus.UNHEALTHY:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report


__all__ = ["router"]
