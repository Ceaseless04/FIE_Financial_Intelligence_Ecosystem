"""HTTP contract: authentication, authorization, validation, and error shape.

The application under test is the real one — real routing, real dependency
graph, real error handlers. Only the stores behind it are stubbed, because what
is asserted is the contract a client sees.

Two properties get particular attention, being the two places an API can undo
the guarantees beneath it:

- **Figures cross the wire as strings.** JSON's only numeric type is a double.
- **Directions cross the wire stated, never implied.** A client left to infer
  "below plan is good here and bad there" from a sign will eventually infer it
  wrong, which is the failure this product is arranged around.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from cfo_fixtures import CHART, actuals, budget, tree

from cfo_ai.analysis.compare import compare
from cfo_ai.api.app import CORRELATION_HEADER, create_app
from cfo_ai.api.dependencies import (
    PERMISSION_READ_COMMENTARY,
    PERMISSION_READ_PLAN,
    PERMISSION_READ_VARIANCE,
    PERMISSION_WRITE_COMMENTARY,
    PERMISSION_WRITE_PLAN,
    get_session,
)
from cfo_ai.config import CFOSettings
from cfo_ai.domain.plan import Plan, PlanKind
from cfo_ai.pipeline import PlanStored, ReportOutcome
from cfo_ai.research.commentary import Commentary, CommentarySection
from fie_auth import AuthSettings, Principal, Role, TokenType, create_token
from fie_common.config import CoreSettings
from fie_common.errors import NotFoundError
from fie_schemas.health import ComponentHealth, HealthStatus
from fie_schemas.provenance import Provenance

pytestmark = pytest.mark.api

AUTH = AuthSettings()

PERIOD_BODY: dict[str, Any] = {
    "kind": "annual",
    "fiscal_year": 2026,
    "start_date": "2026-01-01",
    "end_date": "2026-12-31",
}

PLAN_BODY: dict[str, Any] = {
    "entity_id": "ent-northwind",
    "kind": "budget",
    "period": PERIOD_BODY,
    "currency": "USD",
    "version": "v1",
    "source_id": "bud-fy2026",
    "lines": [
        {
            "account_code": "4000",
            "cost_centre_id": "cc-sales",
            "period": PERIOD_BODY,
            "amount": "4000000",
        }
    ],
}

CHART_BODY: dict[str, Any] = {
    "entity_id": "ent-northwind",
    "accounts": [
        {"code": "4000", "name": "Revenue", "type": "revenue", "aliases": ["sales"]},
        {"code": "6100", "name": "Marketing", "type": "operating_expense"},
        {"code": "9000", "name": "Headcount", "type": "headcount"},
    ],
}


def sample_commentary(*, grounded: bool = True) -> Commentary:
    report = compare(budget(), actuals(), CHART)
    return Commentary(
        entity_id="ent-northwind",
        period_label="FY2026",
        summary="Revenue came in below plan; marketing came in above plan.",
        sections=[CommentarySection(heading="Marketing", body="Above plan.")],
        metrics=report.to_metrics(),
        provenance=Provenance.generated("fake-model-1"),
        numerically_grounded=True,
        directionally_grounded=grounded,
        direction_contradictions=[]
        if grounded
        else ["the narrative describes 6100 as good news, but the variance is unfavourable"],
    )


@dataclass
class StubCharts:
    accounts_by_entity: dict[str, dict] = field(
        default_factory=lambda: {"ent-northwind": dict(CHART)}
    )
    has_tree: bool = True

    async def accounts(self, entity_id: str) -> dict:
        return self.accounts_by_entity.get(entity_id, {})

    async def replace_accounts(self, entity_id: str, accounts) -> int:
        self.accounts_by_entity[entity_id] = {a.code: a for a in accounts}
        return len(accounts)

    async def cost_centres(self, entity_id: str):
        return tree() if self.has_tree else None

    async def replace_cost_centres(self, entity_id: str, value) -> int:
        return len(value.centres)


@dataclass
class StubPlans:
    stored: dict[str, Plan] = field(default_factory=lambda: {"pln-1": budget()})
    present: bool = True

    async def get(self, plan_id: str) -> Plan | None:
        return self.stored.get(plan_id)

    async def list_for_entity(self, entity_id: str, *, kind=None, limit: int = 20):
        return [p for p in self.stored.values() if p.entity_id == entity_id][:limit]

    async def find(self, entity_id: str, *, kind, period, version=None) -> Plan | None:
        if not self.present:
            return None
        return budget() if kind is PlanKind.BUDGET else actuals()

    async def save(self, plan: Plan) -> str:
        return "pln-1"


@dataclass
class StubCommentaries:
    stored: dict[str, Commentary] = field(default_factory=lambda: {"cmt-1": sample_commentary()})

    async def get(self, commentary_id: str):
        item = self.stored.get(commentary_id)
        return (commentary_id, item) if item is not None else None

    async def list_for_entity(self, entity_id: str, *, limit: int = 20, publishable_only=False):
        items = [(k, v) for k, v in self.stored.items() if v.entity_id == entity_id]
        if publishable_only:
            items = [(k, v) for k, v in items if v.is_publishable]
        return items[:limit]

    async def save(self, commentary: Commentary) -> str:
        return "cmt-1"


@dataclass
class StubPlanning:
    unassigned: list[str] = field(default_factory=list)

    async def store_plan(self, plan: Plan) -> PlanStored:
        return PlanStored(
            plan_id="pln-1",
            entity_id=plan.entity_id,
            line_count=len(plan.lines),
            total=str(plan.total().amount),
            unassigned_cost_centres=list(self.unassigned),
        )

    async def publish_chart(self, entity_id: str, accounts) -> int:
        return len(accounts)


@dataclass
class StubReporting:
    commentary: Commentary = field(default_factory=sample_commentary)
    error: Exception | None = None

    async def variance(self, entity_id: str, period, *, budget_version=None):
        if self.error is not None:
            raise self.error
        return compare(budget(), actuals(), CHART)

    async def report(self, entity_id: str, period, **kwargs) -> ReportOutcome:
        if self.error is not None:
            raise self.error
        return ReportOutcome(
            entity_id=entity_id,
            period_label="FY2026",
            variances=compare(budget(), actuals(), CHART),
            commentary_id="cmt-1",
            commentary=self.commentary,
        )


@dataclass
class StubContainer:
    """Presents the ServiceContainer surface the routers actually use."""

    settings: CFOSettings = field(default_factory=CFOSettings)
    core: CoreSettings = field(default_factory=CoreSettings)
    auth: AuthSettings = field(default_factory=AuthSettings)
    chart_store: StubCharts = field(default_factory=StubCharts)
    plan_store: StubPlans = field(default_factory=StubPlans)
    commentary_store: StubCommentaries = field(default_factory=StubCommentaries)
    planning_pipeline: StubPlanning = field(default_factory=StubPlanning)
    reporting_pipeline: StubReporting = field(default_factory=StubReporting)
    components: list[ComponentHealth] = field(
        default_factory=lambda: [
            ComponentHealth(name="postgres", status=HealthStatus.HEALTHY),
            ComponentHealth(name="claude", status=HealthStatus.HEALTHY, required=False),
        ]
    )

    def charts(self, _session: Any) -> StubCharts:
        return self.chart_store

    def plans(self, _session: Any) -> StubPlans:
        return self.plan_store

    def commentaries(self, _session: Any) -> StubCommentaries:
        return self.commentary_store

    def planning(self, _session: Any) -> StubPlanning:
        return self.planning_pipeline

    def reporting(self, _session: Any) -> StubReporting:
        return self.reporting_pipeline

    async def health(self) -> list[ComponentHealth]:
        return list(self.components)

    async def aclose(self) -> None:
        return None


def auth_header(*roles: Role, permissions: list[str] | None = None) -> dict[str, str]:
    principal = Principal(subject="tester", roles=list(roles), permissions=permissions or [])
    return {"Authorization": f"Bearer {create_token(principal, AUTH, token_type=TokenType.ACCESS)}"}


READ_ALL = auth_header(
    permissions=[PERMISSION_READ_PLAN, PERMISSION_READ_VARIANCE, PERMISSION_READ_COMMENTARY]
)
WRITE_PLAN = auth_header(permissions=[PERMISSION_WRITE_PLAN])
WRITE_COMMENTARY = auth_header(permissions=[PERMISSION_WRITE_COMMENTARY])


def flatten_routes(app) -> list:
    """Every real route, walking through included-router wrappers.

    This FastAPI version wraps an included router rather than flattening its
    routes into ``app.routes``. A test that iterates ``app.routes`` looking for
    paths therefore finds none and passes while checking nothing — which is
    exactly what happened to the equivalent Atlas test until this was written.
    """
    found: list = []
    stack = list(app.routes)
    while stack:
        item = stack.pop()
        inner = getattr(item, "original_router", None)
        if inner is not None:
            stack.extend(inner.routes)
            continue
        found.append(item)
    return found


@pytest.fixture
def container() -> StubContainer:
    return StubContainer()


@pytest.fixture
async def client(container: StubContainer) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(container=container, configure_observability=False)  # type: ignore[arg-type]
    app.dependency_overrides[get_session] = lambda: None

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://cfo.test") as http:
            yield http


class TestRoutingTable:
    async def test_every_api_route_declares_authorization(self, container: StubContainer) -> None:
        """An unprotected endpoint should be visible in the routing table rather
        than hiding in a function body — so this asserts on the table, and on
        enough of it to matter."""
        app = create_app(container=container, configure_observability=False)  # type: ignore[arg-type]
        routes = [r for r in flatten_routes(app) if getattr(r, "path", "").startswith("/api/")]

        assert len(routes) >= 8, "the enumeration found too few routes to be meaningful"

        unprotected = [route.path for route in routes if not getattr(route, "dependencies", [])]
        assert unprotected == []


class TestHealth:
    async def test_liveness_needs_no_authentication(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/health/live")
        assert response.status_code == 200

    async def test_an_ai_outage_still_serves_traffic(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        """Every deterministic variance still works without a model. That is
        most of the product and all of the part that has to be right."""
        container.components = [
            ComponentHealth(name="postgres", status=HealthStatus.HEALTHY),
            ComponentHealth(name="claude", status=HealthStatus.UNHEALTHY, required=False),
        ]
        response = await client.get("/health/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "degraded"

    async def test_readiness_fails_when_postgres_is_down(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        container.components = [
            ComponentHealth(name="postgres", status=HealthStatus.UNHEALTHY, required=True)
        ]
        assert (await client.get("/health/ready")).status_code == 503


class TestAuthorization:
    async def test_an_unauthenticated_request_is_401(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/v1/variance?entity_id=ent-northwind", json=PERIOD_BODY)
        assert response.status_code == 401

    async def test_reading_a_variance_does_not_grant_writing_a_budget(
        self, client: httpx.AsyncClient
    ) -> None:
        """The people who may see how a department performed are not always the
        people who set its budget."""
        header = auth_header(permissions=[PERMISSION_READ_VARIANCE])

        allowed = await client.post(
            "/api/v1/variance?entity_id=ent-northwind", json=PERIOD_BODY, headers=header
        )
        refused = await client.post("/api/v1/plans", json=PLAN_BODY, headers=header)

        assert allowed.status_code == 200
        assert refused.status_code == 403

    async def test_writing_a_plan_does_not_grant_generating_commentary(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.post(
            "/api/v1/commentary",
            json={"entity_id": "ent-northwind", "period": PERIOD_BODY},
            headers=WRITE_PLAN,
        )
        assert response.status_code == 403


class TestPlans:
    async def test_submitting_a_plan_returns_201(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/v1/plans", json=PLAN_BODY, headers=WRITE_PLAN)

        assert response.status_code == 201
        assert response.json()["plan_id"] == "pln-1"

    async def test_a_float_amount_is_422_not_500(self, client: httpx.AsyncClient) -> None:
        """A JSON number is already a float by the time Pydantic sees it."""
        body = {**PLAN_BODY, "lines": [{**PLAN_BODY["lines"][0], "amount": 4000000.5}]}
        response = await client.post("/api/v1/plans", json=body, headers=WRITE_PLAN)
        assert response.status_code == 422

    async def test_a_duplicate_line_is_refused_at_the_edge(self, client: httpx.AsyncClient) -> None:
        """A second row for one key double-counts into every total built from it."""
        body = {**PLAN_BODY, "lines": [PLAN_BODY["lines"][0], PLAN_BODY["lines"][0]]}
        response = await client.post("/api/v1/plans", json=body, headers=WRITE_PLAN)
        assert response.status_code == 422

    async def test_unassigned_cost_centres_are_returned_not_hidden(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        """Spend in a centre the org chart does not know about rolls up nowhere."""
        container.planning_pipeline.unassigned = ["cc-newteam"]

        response = await client.post("/api/v1/plans", json=PLAN_BODY, headers=WRITE_PLAN)

        assert response.json()["unassigned_cost_centres"] == ["cc-newteam"]

    async def test_plan_amounts_come_back_as_strings(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/plans/pln-1", headers=READ_ALL)
        body = response.json()

        assert isinstance(body["total"], str)
        assert all(isinstance(line["amount"], str) for line in body["lines"])

    async def test_an_unknown_plan_is_404(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/plans/pln-missing", headers=READ_ALL)
        assert response.status_code == 404


class TestChartAndOrganisation:
    async def test_publishing_a_chart_echoes_the_direction_rules(
        self, client: httpx.AsyncClient
    ) -> None:
        """So a client never re-derives "is higher better here" for itself."""
        response = await client.put(
            "/api/v1/entities/ent-northwind/chart", json=CHART_BODY, headers=WRITE_PLAN
        )
        body = {item["code"]: item for item in response.json()}

        assert response.status_code == 200
        assert body["4000"]["higher_is_better"] is True
        assert body["6100"]["higher_is_better"] is False
        assert body["9000"]["has_direction"] is False

    async def test_a_mismatched_entity_is_422(self, client: httpx.AsyncClient) -> None:
        response = await client.put(
            "/api/v1/entities/ent-other/chart", json=CHART_BODY, headers=WRITE_PLAN
        )
        assert response.status_code == 422

    async def test_a_cyclic_hierarchy_is_422_not_500(self, client: httpx.AsyncClient) -> None:
        """A cycle would make every roll-up built from the tree non-terminating."""
        response = await client.put(
            "/api/v1/entities/ent-northwind/organisation",
            json={
                "entity_id": "ent-northwind",
                "centres": [
                    {"id": "a", "name": "A", "parent_id": "b"},
                    {"id": "b", "name": "B", "parent_id": "a"},
                ],
            },
            headers=WRITE_PLAN,
        )
        assert response.status_code == 422

    async def test_a_valid_hierarchy_is_accepted(self, client: httpx.AsyncClient) -> None:
        response = await client.put(
            "/api/v1/entities/ent-northwind/organisation",
            json={
                "entity_id": "ent-northwind",
                "centres": [
                    {"id": "cc-co", "name": "Company"},
                    {"id": "cc-eng", "name": "Engineering", "parent_id": "cc-co"},
                ],
            },
            headers=WRITE_PLAN,
        )
        assert response.status_code == 204


class TestVarianceContract:
    async def test_every_figure_is_a_string(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/variance?entity_id=ent-northwind", json=PERIOD_BODY, headers=READ_ALL
        )
        body = response.json()

        assert response.status_code == 200
        assert isinstance(body["net_operating_variance"], str)
        assert all(isinstance(item["amount"], str) for item in body["variances"])

    async def test_each_line_carries_its_direction(self, client: httpx.AsyncClient) -> None:
        """Stated, not implied. A client inferring it from the sign gets it
        backwards on costs sooner or later."""
        response = await client.post(
            "/api/v1/variance?entity_id=ent-northwind", json=PERIOD_BODY, headers=READ_ALL
        )
        by_code = {item["account_code"]: item for item in response.json()["variances"]}

        # Revenue 4.0m planned, 3.6m actual — below plan, and bad.
        assert by_code["4000"]["favourability"] == "unfavourable"
        # Marketing 600k planned, 700k actual — above plan, and also bad.
        assert by_code["6100"]["favourability"] == "unfavourable"
        # COGS came in under plan — below, and good.
        assert by_code["5000"]["favourability"] == "favourable"
        assert "unfavourable" in by_code["4000"]["description"]

    async def test_the_directions_are_bucketed_for_the_client(
        self, client: httpx.AsyncClient
    ) -> None:
        """Bucketing client-side would mean re-deriving the rule this service
        exists to decide once."""
        body = (
            await client.post(
                "/api/v1/variance?entity_id=ent-northwind", json=PERIOD_BODY, headers=READ_ALL
            )
        ).json()

        assert {"variances", "material", "unfavourable", "favourable", "not_assessed"} <= set(body)
        assert len(body["unfavourable"]) == 2
        assert len(body["favourable"]) == 2

    async def test_rollups_are_included_by_default(self, client: httpx.AsyncClient) -> None:
        body = (
            await client.post(
                "/api/v1/variance?entity_id=ent-northwind", json=PERIOD_BODY, headers=READ_ALL
            )
        ).json()

        assert body["rollups"]
        assert body["rollups"][0]["cost_centre_id"] == "cc-co"

    async def test_a_missing_plan_is_404(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        container.reporting_pipeline.error = NotFoundError("no budget is on file")

        response = await client.post(
            "/api/v1/variance?entity_id=ent-northwind", json=PERIOD_BODY, headers=READ_ALL
        )
        assert response.status_code == 404

    async def test_an_invalid_period_is_422(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/variance?entity_id=ent-northwind",
            json={"kind": "quarterly", "fiscal_year": 2026, "end_date": "2026-03-31"},
            headers=READ_ALL,
        )
        assert response.status_code == 422


class TestCommentaryContract:
    async def test_a_verified_commentary_is_publishable(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/commentary",
            json={"entity_id": "ent-northwind", "period": PERIOD_BODY},
            headers=WRITE_COMMENTARY,
        )
        body = response.json()

        assert response.status_code == 201
        assert body["publishable"] is True
        assert body["directionally_grounded"] is True

    async def test_a_reversed_commentary_is_returned_labelled(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        """Every figure can be real and the conclusion backwards. Suppressing it
        would hide a systematic model problem; a client that checks the flag can
        never mistake it for analysis."""
        container.reporting_pipeline.commentary = sample_commentary(grounded=False)

        response = await client.post(
            "/api/v1/commentary",
            json={"entity_id": "ent-northwind", "period": PERIOD_BODY},
            headers=WRITE_COMMENTARY,
        )
        body = response.json()

        assert response.status_code == 201
        assert body["publishable"] is False
        assert body["numerically_grounded"] is True
        assert body["direction_contradictions"]

    async def test_the_metrics_are_never_attributed_to_the_model(
        self, client: httpx.AsyncClient
    ) -> None:
        body = (
            await client.post(
                "/api/v1/commentary",
                json={"entity_id": "ent-northwind", "period": PERIOD_BODY},
                headers=WRITE_COMMENTARY,
            )
        ).json()

        assert body["provenance"]["model"] == "fake-model-1"
        assert all(metric["provenance"].get("model") is None for metric in body["metrics"])

    async def test_metric_values_are_strings(self, client: httpx.AsyncClient) -> None:
        body = (
            await client.post(
                "/api/v1/commentary",
                json={"entity_id": "ent-northwind", "period": PERIOD_BODY},
                headers=WRITE_COMMENTARY,
            )
        ).json()
        assert all(isinstance(metric["value"], str) for metric in body["metrics"])

    async def test_listing_returns_reversed_commentary_by_default(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        container.commentary_store.stored = {
            "cmt-1": sample_commentary(),
            "cmt-2": sample_commentary(grounded=False),
        }

        every = await client.get("/api/v1/commentary/entities/ent-northwind", headers=READ_ALL)
        filtered = await client.get(
            "/api/v1/commentary/entities/ent-northwind?publishable_only=true", headers=READ_ALL
        )

        assert len(every.json()) == 2
        assert len(filtered.json()) == 1

    async def test_an_unknown_commentary_is_404(self, client: httpx.AsyncClient) -> None:
        assert (
            await client.get("/api/v1/commentary/cmt-missing", headers=READ_ALL)
        ).status_code == 404


class TestErrorEnvelope:
    async def test_a_correlation_id_is_echoed(self, client: httpx.AsyncClient) -> None:
        response = await client.get(
            "/health/live", headers={CORRELATION_HEADER: "corr-from-caller"}
        )
        assert response.headers[CORRELATION_HEADER] == "corr-from-caller"

    async def test_an_unhandled_error_leaks_nothing_but_keeps_the_id(
        self, container: StubContainer
    ) -> None:
        container.reporting_pipeline.error = RuntimeError("dsn: postgres://secret")
        app = create_app(container=container, configure_observability=False)  # type: ignore[arg-type]
        app.dependency_overrides[get_session] = lambda: None

        # Starlette generates the 500 body and then re-raises so the server can
        # log it; here that has to be suppressed to inspect what the client got.
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://cfo.test") as c:
                response = await c.post(
                    "/api/v1/variance?entity_id=ent-northwind",
                    json=PERIOD_BODY,
                    headers=READ_ALL | {CORRELATION_HEADER: "corr-trace-me"},
                )

        body = response.json()
        assert response.status_code == 500
        assert "secret" not in response.text
        assert body["correlation_id"] == "corr-trace-me"

    async def test_a_validation_error_names_the_offending_field(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.post(
            "/api/v1/plans", json={**PLAN_BODY, "entity_id": ""}, headers=WRITE_PLAN
        )
        body = response.json()

        assert response.status_code == 422
        assert body["code"] == "validation_error"
        assert body["details"]["errors"]
