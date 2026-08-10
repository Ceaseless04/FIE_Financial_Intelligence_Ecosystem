"""Health, readiness, and liveness contracts (Phase 9 reliability requirement)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from fie_common.utils import utc_now
from fie_schemas.base import FIEModel, FrozenModel


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

    @property
    def is_serving(self) -> bool:
        """Degraded still serves traffic — graceful degradation, not an outage."""
        return self is not HealthStatus.UNHEALTHY


class ComponentHealth(FrozenModel):
    """Health of one dependency (Postgres, Redis, Neo4j, an AI provider...)."""

    name: str = Field(min_length=1)
    status: HealthStatus
    latency_ms: float | None = Field(default=None, ge=0)
    message: str | None = None
    #: Whether the service can serve traffic at all without this component.
    required: bool = True


class HealthReport(FIEModel):
    """Aggregate health of a service and its dependencies."""

    service: str
    version: str
    environment: str
    status: HealthStatus = HealthStatus.HEALTHY
    components: list[ComponentHealth] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def from_components(
        cls,
        *,
        service: str,
        version: str,
        environment: str,
        components: list[ComponentHealth],
    ) -> HealthReport:
        """Roll component health up to a service verdict.

        A failed *required* component makes the service unhealthy; a failed
        optional one only degrades it. That distinction is what lets the
        ecosystem degrade gracefully instead of cascading — a Neo4j outage
        should not take Atlas offline if Atlas can still serve filings.
        """
        status = HealthStatus.HEALTHY
        for component in components:
            if component.status is HealthStatus.UNHEALTHY:
                if component.required:
                    status = HealthStatus.UNHEALTHY
                    break
                status = HealthStatus.DEGRADED
            elif component.status is HealthStatus.DEGRADED and status is HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED
        return cls(
            service=service,
            version=version,
            environment=environment,
            status=status,
            components=components,
        )

    @property
    def is_ready(self) -> bool:
        """Readiness: every required dependency is reachable."""
        return all(
            component.status.is_serving for component in self.components if component.required
        )


__all__ = ["ComponentHealth", "HealthReport", "HealthStatus"]
