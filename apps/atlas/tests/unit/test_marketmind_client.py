"""The MarketMind client, and the degradation rule it exists to enforce.

Graph context makes a research note better. It is never what a figure is
computed from — so MarketMind being down must cost Atlas the context and nothing
else. A report about figures Atlas already computed and verified must not fail
because a different service was unreachable.

The transport is intercepted rather than stubbed at the client's own methods:
what is being asserted is how real HTTP failures behave, and a fake client that
returns what the test told it to would assert nothing.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from atlas.clients.marketmind import CORRELATION_HEADER, MarketMindClient
from fie_common.errors import NotFoundError
from fie_common.resilience import RetryPolicy
from fie_observability.context import request_context
from fie_schemas.health import HealthStatus
from fie_schemas.provenance import SourceType

pytestmark = pytest.mark.unit

BASE_URL = "http://marketmind.test"

ENTITY_BODY = {
    "id": "ent-acme",
    "name": "Acme Robotics Corporation",
    "canonical_name": "acme robotics",
    "aliases": ["Acme"],
    "identifiers": {"ticker": "ACME", "cik": "123456"},
    "description": "Industrial robotics manufacturer.",
    "source_document_ids": ["doc-1"],
}

NEIGHBOURS_BODY = {
    "edges": [
        {
            "type": "SUPPLIES",
            "source_name": "Northwind Components Ltd",
            "target_name": "Acme Robotics Corporation",
            "source_id": "ent-northwind",
            "target_id": "ent-acme",
        }
    ]
}


def client(**overrides: object) -> MarketMindClient:
    return MarketMindClient(
        base_url=BASE_URL,
        timeout_seconds=1.0,
        # No backoff: these tests assert retry *behaviour*, not wall time.
        retry=RetryPolicy(max_attempts=2, initial_backoff_seconds=0.0, jitter=0.0),
        **overrides,  # type: ignore[arg-type]
    )


class TestCompanyLookup:
    @respx.mock
    async def test_an_entity_is_read_from_the_graph(self) -> None:
        respx.get(f"{BASE_URL}/api/v1/entities/ent-acme").mock(
            return_value=httpx.Response(200, json=ENTITY_BODY)
        )
        subject = client()

        company = await subject.company("ent-acme")
        await subject.aclose()

        assert company is not None
        assert company.name == "Acme Robotics Corporation"
        assert company.identifiers["ticker"] == "ACME"

    @respx.mock
    async def test_an_unknown_entity_raises_rather_than_degrading(self) -> None:
        """A 404 is an answer, not an outage. MarketMind is authoritative about
        whether an entity exists, and Atlas will not invent one."""
        respx.get(f"{BASE_URL}/api/v1/entities/ent-nobody").mock(return_value=httpx.Response(404))
        subject = client()

        with pytest.raises(NotFoundError):
            await subject.company("ent-nobody")
        await subject.aclose()

    @respx.mock
    async def test_an_outage_returns_none_rather_than_raising(self) -> None:
        respx.get(f"{BASE_URL}/api/v1/entities/ent-acme").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        subject = client()

        assert await subject.company("ent-acme") is None
        await subject.aclose()

    @respx.mock
    async def test_graph_context_is_citable_as_the_graph(self) -> None:
        """A claim drawn from the graph should not appear to come from a filing."""
        respx.get(f"{BASE_URL}/api/v1/entities/ent-acme").mock(
            return_value=httpx.Response(200, json=ENTITY_BODY)
        )
        subject = client()

        company = await subject.company("ent-acme")
        await subject.aclose()

        assert company is not None
        reference = company.to_source_reference()
        assert reference.source_type is SourceType.KNOWLEDGE_GRAPH
        assert reference.source_id == "ent-acme"


class TestContextDegradation:
    @respx.mock
    async def test_related_companies_become_sentences(self) -> None:
        respx.get(f"{BASE_URL}/api/v1/graph/entities/ent-acme/neighbours").mock(
            return_value=httpx.Response(200, json=NEIGHBOURS_BODY)
        )
        subject = client()

        facts = await subject.context_for("ent-acme")
        await subject.aclose()

        assert facts == ["Northwind Components Ltd — supplies — Acme Robotics Corporation"]

    @respx.mock
    @pytest.mark.parametrize(
        "failure",
        [
            httpx.Response(500),
            httpx.Response(404),
            httpx.Response(401),
        ],
    )
    async def test_any_failure_costs_only_the_context(self, failure: httpx.Response) -> None:
        """Including a 404: Atlas routinely analyses a filing for a company the
        graph has not ingested yet, and that is a report without context."""
        respx.get(f"{BASE_URL}/api/v1/graph/entities/ent-acme/neighbours").mock(
            return_value=failure
        )
        subject = client()

        assert await subject.context_for("ent-acme") == []
        await subject.aclose()

    @respx.mock
    async def test_a_transport_failure_costs_only_the_context(self) -> None:
        respx.get(f"{BASE_URL}/api/v1/graph/entities/ent-acme/neighbours").mock(
            side_effect=httpx.ReadTimeout("timed out")
        )
        subject = client()

        assert await subject.context_for("ent-acme") == []
        await subject.aclose()


class TestRetryPolicy:
    @respx.mock
    async def test_a_server_error_is_retried(self) -> None:
        route = respx.get(f"{BASE_URL}/api/v1/entities/ent-acme").mock(
            side_effect=[httpx.Response(503), httpx.Response(200, json=ENTITY_BODY)]
        )
        subject = client()

        company = await subject.company("ent-acme")
        await subject.aclose()

        assert company is not None
        assert route.call_count == 2

    @respx.mock
    async def test_a_client_error_is_not_retried(self) -> None:
        """A 401 from a missing service token does not become correct by being
        repeated; retrying only multiplies the latency before giving up."""
        route = respx.get(f"{BASE_URL}/api/v1/entities/ent-acme").mock(
            return_value=httpx.Response(401)
        )
        subject = client()

        assert await subject.company("ent-acme") is None
        await subject.aclose()
        assert route.call_count == 1


class TestObservabilityAndAuth:
    @respx.mock
    async def test_a_service_token_is_sent(self) -> None:
        route = respx.get(f"{BASE_URL}/api/v1/entities/ent-acme").mock(
            return_value=httpx.Response(200, json=ENTITY_BODY)
        )
        subject = client(token="service-token")

        await subject.company("ent-acme")
        await subject.aclose()

        assert route.calls[0].request.headers["authorization"] == "Bearer service-token"

    @respx.mock
    async def test_the_correlation_id_crosses_the_service_boundary(self) -> None:
        """Without it, a slow report and the graph query behind it cannot be joined."""
        route = respx.get(f"{BASE_URL}/api/v1/entities/ent-acme").mock(
            return_value=httpx.Response(200, json=ENTITY_BODY)
        )
        subject = client()

        with request_context(correlation_id="corr-abc", service_name="atlas"):
            await subject.company("ent-acme")
        await subject.aclose()

        assert route.calls[0].request.headers[CORRELATION_HEADER] == "corr-abc"

    @respx.mock
    async def test_health_reports_rather_than_raises(self) -> None:
        respx.get(f"{BASE_URL}/health/live").mock(side_effect=httpx.ConnectError("down"))
        subject = client()

        health = await subject.health_check()
        await subject.aclose()

        assert health.status is HealthStatus.UNHEALTHY
        # Non-required: Atlas computes and verifies every figure without
        # MarketMind, so an outage here must not take Atlas out of rotation.
        assert health.required is False

    @respx.mock
    async def test_a_healthy_marketmind_is_reported_healthy(self) -> None:
        respx.get(f"{BASE_URL}/health/live").mock(
            return_value=httpx.Response(200, json={"status": "alive"})
        )
        subject = client()

        health = await subject.health_check()
        await subject.aclose()

        assert health.status is HealthStatus.HEALTHY
