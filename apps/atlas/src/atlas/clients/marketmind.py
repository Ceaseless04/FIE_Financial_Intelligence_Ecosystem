"""Client for MarketMind's knowledge graph.

Atlas asks MarketMind two things: who is this entity, and what is it connected
to. That context makes a research note better — a margin decline reads
differently when the graph shows the company's largest customer was acquired by
a competitor — but it is never what a figure is computed from.

**Every failure here degrades to no context, never to no report.** MarketMind
being slow or down is an infrastructure event; refusing to write a report about
figures Atlas already computed and verified would turn it into an analytical
one. The report simply carries no graph context, and says so.

The one thing that is *not* degraded is identity. Atlas does not fall back to
inventing an entity, or to matching on name, when MarketMind cannot confirm one
— duplicating resolution here is how two products end up disagreeing about who
a company is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import Field

from fie_ai.http import build_async_client
from fie_common.errors import ExternalServiceError, NotFoundError
from fie_common.resilience import RetryPolicy, default_retryable, retry_async
from fie_observability.context import current_context
from fie_observability.logging import get_logger
from fie_schemas.base import FrozenModel
from fie_schemas.health import ComponentHealth, HealthStatus
from fie_schemas.provenance import SourceReference, SourceType

logger = get_logger(__name__)

CORRELATION_HEADER = "X-Correlation-ID"


def _is_retryable(error: BaseException) -> bool:
    """Retry transport failures and server errors, never a 4xx.

    A 401 from a missing service token does not become correct by being
    repeated; retrying it only multiplies the latency a report pays before
    giving up on context it was always going to do without.
    """
    if isinstance(error, ExternalServiceError):
        status = error.details.get("status")
        return not (isinstance(status, int) and 400 <= status < 500)
    return default_retryable(error)


class GraphCompany(FrozenModel):
    """What MarketMind knows about a company."""

    entity_id: str
    name: str
    canonical_name: str = ""
    aliases: list[str] = Field(default_factory=list)
    identifiers: dict[str, str] = Field(default_factory=dict)
    description: str | None = None
    source_document_ids: list[str] = Field(default_factory=list)

    def to_source_reference(self) -> SourceReference:
        """A citable pointer to the graph itself.

        Graph context is evidence like any other, and a claim drawn from it
        should be attributable to the graph rather than appearing to come from
        the filing.
        """
        return SourceReference(
            source_id=self.entity_id,
            source_type=SourceType.KNOWLEDGE_GRAPH,
            title=self.name,
        )


@dataclass
class MarketMindClient:
    """Reads company context from MarketMind over HTTP."""

    base_url: str
    token: str | None = None
    timeout_seconds: float = 10.0
    #: Two attempts, not five. Context is optional, and a caller waiting on a
    #: long retry chain for something the report can do without is a worse
    #: outcome than no context.
    retry: RetryPolicy = field(
        default_factory=lambda: RetryPolicy(max_attempts=2, initial_backoff_seconds=0.2)
    )
    _client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            self._client = build_async_client(
                base_url=self.base_url, timeout=self.timeout_seconds, headers=headers
            )
        return self._client

    async def company(self, entity_id: str) -> GraphCompany | None:
        """Look up an entity. ``None`` when MarketMind is unreachable.

        A 404 is a different answer from an outage and is *not* swallowed: the
        caller asked about a specific entity id and MarketMind is authoritative
        about whether it exists.

        Raises:
            NotFoundError: if MarketMind does not know this entity.
        """
        try:
            payload = await self._get(f"/api/v1/entities/{entity_id}")
        except NotFoundError:
            raise
        except ExternalServiceError as error:
            logger.warning("marketmind_unavailable", entity_id=entity_id, error=str(error))
            return None

        return GraphCompany(
            entity_id=str(payload.get("id", entity_id)),
            name=str(payload.get("name", "")),
            canonical_name=str(payload.get("canonical_name") or ""),
            aliases=[str(alias) for alias in payload.get("aliases", [])],
            identifiers={
                str(key): str(value) for key, value in (payload.get("identifiers") or {}).items()
            },
            description=payload.get("description"),
            source_document_ids=[str(doc) for doc in payload.get("source_document_ids", [])],
        )

    async def context_for(self, entity_id: str, *, limit: int = 12) -> list[str]:
        """Related-company facts, as sentences a narrative can use.

        Returns an empty list on any failure — including a 404. Here, unlike
        :meth:`company`, an unknown entity is not exceptional: Atlas routinely
        analyses a filing for a company the graph has not ingested yet, and that
        is a report without context, not an error.
        """
        try:
            payload = await self._get(
                f"/api/v1/graph/entities/{entity_id}/neighbours", params={"limit": limit}
            )
        except (ExternalServiceError, NotFoundError) as error:
            logger.info("marketmind_context_unavailable", entity_id=entity_id, error=str(error))
            return []

        facts: list[str] = []
        for edge in payload.get("edges", [])[:limit]:
            source = edge.get("source_name") or edge.get("source_id")
            target = edge.get("target_name") or edge.get("target_id")
            relationship = str(edge.get("type", "related to")).replace("_", " ").lower()
            if source and target:
                facts.append(f"{source} — {relationship} — {target}")
        return facts

    async def health_check(self) -> ComponentHealth:
        """MarketMind's reachability, reported as a non-required dependency.

        Non-required is the whole point: Atlas computes and verifies figures
        without MarketMind, so an outage here must not take Atlas out of
        rotation.
        """
        try:
            await self._get("/health/live")
        except Exception as error:  # noqa: BLE001 — a health check reports, never raises
            return ComponentHealth(
                name="marketmind",
                status=HealthStatus.UNHEALTHY,
                required=False,
                message=str(error)[:200],
            )
        return ComponentHealth(name="marketmind", status=HealthStatus.HEALTHY, required=False)

    async def _get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """One GET, with retries and error translation.

        Raises:
            NotFoundError: on a 404.
            ExternalServiceError: on any other failure, including transport
                errors and non-2xx responses.
        """

        async def call() -> dict[str, Any]:
            headers = {}
            correlation_id = current_context().correlation_id
            if correlation_id:
                # Carried so one id spans Atlas and MarketMind; without it a
                # slow report and the graph query behind it cannot be joined.
                headers[CORRELATION_HEADER] = correlation_id
            try:
                response = await self._http().get(path, params=params, headers=headers)
            except httpx.HTTPError as error:
                raise ExternalServiceError(
                    f"marketmind request failed: {error}", details={"path": path}
                ) from error

            if response.status_code == 404:
                raise NotFoundError("marketmind does not know this entity", details={"path": path})
            if response.status_code >= 400:
                raise ExternalServiceError(
                    f"marketmind returned {response.status_code}",
                    details={"path": path, "status": response.status_code},
                )

            body = response.json()
            if not isinstance(body, dict):
                raise ExternalServiceError(
                    "marketmind returned a non-object body", details={"path": path}
                )
            return body

        return await retry_async(call, policy=self.retry, retryable=_is_retryable)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


__all__ = ["CORRELATION_HEADER", "GraphCompany", "MarketMindClient"]
