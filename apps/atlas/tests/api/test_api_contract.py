"""HTTP contract: authentication, authorization, validation, and error shape.

The application under test is the real one — real routing, real dependency
graph, real error handlers. Only the stores behind it are stubbed, because what
is being asserted is the contract a client sees, and a client cannot tell
whether Postgres was involved in a 403.

Two contract properties get particular attention here, because both are places
where an API can quietly undo the guarantees underneath it:

- **Figures cross the wire as strings.** JSON's only numeric type is a double.
- **An ungrounded report is still returned**, labelled, rather than suppressed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import httpx
import pytest
from atlas_fixtures import statements as build_statements

from atlas.analysis.ratios import analyze
from atlas.api.app import CORRELATION_HEADER, create_app
from atlas.api.dependencies import (
    PERMISSION_READ_ANALYSIS,
    PERMISSION_READ_FILING,
    PERMISSION_READ_RESEARCH,
    PERMISSION_READ_VALUATION,
    PERMISSION_WRITE_FILING,
    PERMISSION_WRITE_RESEARCH,
    get_session,
)
from atlas.config import AtlasSettings
from atlas.filings.models import Filing
from atlas.filings.pipeline import IngestionResult
from atlas.research.pipeline import ReportOutcome
from atlas.research.reports import ReportSection, ResearchReport
from fie_auth import AuthSettings, Principal, Role, TokenType, create_token
from fie_common.config import CoreSettings
from fie_common.errors import NotFoundError
from fie_finance.periods import PeriodKind
from fie_schemas.health import ComponentHealth, HealthStatus
from fie_schemas.provenance import Provenance

pytestmark = pytest.mark.api

AUTH = AuthSettings()

FILING_BODY: dict[str, Any] = {
    "entity_id": "ent-acme",
    "type": "10-K",
    "period": {"kind": "annual", "fiscal_year": 2025, "end_date": "2025-12-31"},
    "title": "Acme Robotics Corporation Form 10-K FY2025",
    "content": "Item 8. Financial Statements. Revenue was 412.6 million.",
}

VALUATION_BODY: dict[str, Any] = {
    "entity_id": "ent-acme",
    "discount_rate": "0.10",
    "terminal_growth": "0.025",
    "forecast_growth": "0.05",
    "projection_years": 5,
}


def sample_filing() -> Filing:
    from atlas_fixtures import filing as build_filing

    return build_filing()


def sample_report(*, grounded: bool = True) -> ResearchReport:
    analysis = analyze(build_statements())
    return ResearchReport(
        entity_id="ent-acme",
        period_label="FY2025",
        summary="Net margin was 23.4% on revenue of $412.6 million [fil-acme-2025].",
        sections=[ReportSection(heading="Margins", body="Gross margin held at 40.0%.")],
        open_questions=["What drove the effective tax rate?"],
        metrics=analysis.available,
        cited_source_ids=["fil-acme-2025"],
        provenance=Provenance.generated("fake-model-1"),
        numerically_grounded=grounded,
        unsupported_figures=[] if grounded else ["31.2%"],
        citations_grounded=True,
        regeneration_count=0 if grounded else 1,
    )


@dataclass
class StubFilings:
    stored: dict[str, Filing] = field(default_factory=lambda: {"fil-1": sample_filing()})

    async def get(self, filing_id: str) -> Filing | None:
        return self.stored.get(filing_id)

    async def list_for_entity(self, entity_id: str, *, limit: int = 20) -> list[Filing]:
        return [f for f in self.stored.values() if f.entity_id == entity_id][:limit]


@dataclass
class StubStatements:
    present: bool = True

    async def latest(self, entity_id: str) -> Any:
        return build_statements() if self.present else None

    async def get_for_period(
        self, entity_id: str, *, fiscal_year: int, kind: PeriodKind, quarter: int | None = None
    ) -> Any:
        return build_statements() if self.present else None


@dataclass
class StubReports:
    stored: dict[str, ResearchReport] = field(default_factory=lambda: {"rpt-1": sample_report()})

    async def get(self, report_id: str) -> tuple[str, ResearchReport] | None:
        report = self.stored.get(report_id)
        return (report_id, report) if report is not None else None

    async def list_for_entity(
        self, entity_id: str, *, limit: int = 20, publishable_only: bool = False
    ) -> list[tuple[str, ResearchReport]]:
        items = [(rid, r) for rid, r in self.stored.items() if r.entity_id == entity_id]
        if publishable_only:
            items = [(rid, r) for rid, r in items if r.is_publishable]
        return items[:limit]


@dataclass
class StubIngestion:
    result: IngestionResult | None = None

    async def ingest(self, filing: Filing) -> IngestionResult:
        return self.result or IngestionResult(
            filing_id="fil-1",
            entity_id=filing.entity_id,
            period_label=filing.period.label,
            statement_set_id="sts-1",
            has_income_statement=True,
            has_balance_sheet=True,
            has_cash_flow_statement=True,
        )


@dataclass
class StubResearch:
    report: ResearchReport = field(default_factory=sample_report)
    error: Exception | None = None
    valuation: Any = None

    async def produce(self, entity_id: str, **_kwargs: Any) -> ReportOutcome:
        if self.error is not None:
            raise self.error
        return ReportOutcome(
            report_id="rpt-1", report=self.report, analysis=analyze(build_statements())
        )

    async def load_statements(self, entity_id: str, fiscal_year: int | None = None) -> Any:
        if self.error is not None:
            raise self.error
        return build_statements()

    def value(self, entity_id: str, statements: Any, inputs: Any) -> Any:
        from atlas.analysis.dcf import DCFAssumptions, DCFModel
        from atlas.analysis.growth import free_cash_flow
        from fie_finance.money import Money

        if self.valuation == "unavailable":
            return None
        base = free_cash_flow(statements)
        model = DCFModel(
            assumptions=DCFAssumptions(
                discount_rate=inputs.discount_rate,
                terminal_growth_rate=inputs.terminal_growth,
                projection_years=inputs.projection_years,
            ),
            sources=statements.all_sources(),
        )
        flows = [Money(amount=base.value, currency="USD")] * inputs.projection_years
        return model.value(entity_id, flows)


@dataclass
class StubContainer:
    """Presents the ServiceContainer surface the routers actually use."""

    settings: AtlasSettings = field(default_factory=AtlasSettings)
    core: CoreSettings = field(default_factory=CoreSettings)
    auth: AuthSettings = field(default_factory=AuthSettings)
    filing_store: StubFilings = field(default_factory=StubFilings)
    statement_store: StubStatements = field(default_factory=StubStatements)
    report_repository: StubReports = field(default_factory=StubReports)
    ingestion_pipeline: StubIngestion = field(default_factory=StubIngestion)
    research_pipeline: StubResearch = field(default_factory=StubResearch)
    #: Read by the startup log line. None is the honest value here: these tests
    #: never reach MarketMind, and a stub would imply they did.
    marketmind: None = None
    components: list[ComponentHealth] = field(
        default_factory=lambda: [
            ComponentHealth(name="postgres", status=HealthStatus.HEALTHY),
            ComponentHealth(name="claude", status=HealthStatus.HEALTHY, required=False),
            ComponentHealth(name="marketmind", status=HealthStatus.HEALTHY, required=False),
        ]
    )

    def filings(self, _session: Any) -> StubFilings:
        return self.filing_store

    def statements(self, _session: Any) -> StubStatements:
        return self.statement_store

    def report_store(self, _session: Any) -> StubReports:
        return self.report_repository

    def ingestion(self, _session: Any) -> StubIngestion:
        return self.ingestion_pipeline

    def research(self, _session: Any) -> StubResearch:
        return self.research_pipeline

    async def health(self) -> list[ComponentHealth]:
        return list(self.components)

    async def aclose(self) -> None:
        return None


def token_for(*roles: Role, permissions: list[str] | None = None) -> str:
    principal = Principal(subject="tester", roles=list(roles), permissions=permissions or [])
    return create_token(principal, AUTH, token_type=TokenType.ACCESS)


def auth_header(*roles: Role, permissions: list[str] | None = None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token_for(*roles, permissions=permissions)}"}


READ_ALL = auth_header(
    permissions=[
        PERMISSION_READ_FILING,
        PERMISSION_READ_ANALYSIS,
        PERMISSION_READ_VALUATION,
        PERMISSION_READ_RESEARCH,
    ]
)
WRITE_FILING = auth_header(permissions=[PERMISSION_WRITE_FILING])
WRITE_RESEARCH = auth_header(permissions=[PERMISSION_WRITE_RESEARCH])


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
            transport=transport, base_url="http://atlas.test"
        ) as http_client:
            yield http_client


class TestHealth:
    async def test_liveness_needs_no_authentication(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}

    async def test_an_ai_outage_still_serves_traffic(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        """Every deterministic figure still works without a model. That is most
        of the product, and taking the instance out of rotation would lose it."""
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

        response = await client.get("/health/ready")

        assert response.status_code == 503


class TestAuthentication:
    async def test_an_unauthenticated_request_is_401(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/entities/ent-acme/analysis")
        assert response.status_code == 401
        assert response.json()["code"] == "authentication_error"

    async def test_a_token_without_the_permission_is_403(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/filings", json=FILING_BODY, headers=auth_header(permissions=["atlas:other"])
        )
        assert response.status_code == 403

    async def test_reading_analysis_does_not_grant_valuation(
        self, client: httpx.AsyncClient
    ) -> None:
        """A ratio is a fact about a filing; a DCF is an estimate that reads like
        one. An organisation may reasonably separate who sees which."""
        header = auth_header(permissions=[PERMISSION_READ_ANALYSIS])

        analysis = await client.get("/api/v1/entities/ent-acme/analysis", headers=header)
        valuation = await client.post("/api/v1/valuation/dcf", json=VALUATION_BODY, headers=header)

        assert analysis.status_code == 200
        assert valuation.status_code == 403

    async def test_a_malformed_token_is_401_not_500(self, client: httpx.AsyncClient) -> None:
        response = await client.get(
            "/api/v1/entities/ent-acme/analysis", headers={"Authorization": "Bearer not-a-jwt"}
        )
        assert response.status_code == 401


class TestFilings:
    async def test_ingesting_a_filing_returns_201(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/v1/filings", json=FILING_BODY, headers=WRITE_FILING)

        assert response.status_code == 201
        body = response.json()
        assert body["statement_set_id"] == "sts-1"
        assert body["newly_ingested"] is True

    async def test_a_filing_whose_statements_failed_still_returns_201(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        """The document is stored either way; losing it would be the worse outcome."""
        container.ingestion_pipeline.result = IngestionResult(
            filing_id="fil-1",
            entity_id="ent-acme",
            period_label="FY2025",
            rejected=["balance sheet rejected: balance sheet does not balance"],
            failed_validation=True,
        )

        response = await client.post("/api/v1/filings", json=FILING_BODY, headers=WRITE_FILING)

        assert response.status_code == 201
        body = response.json()
        assert body["statement_set_id"] is None
        assert body["failed_validation"] is True
        assert "does not balance" in body["rejected"][0]

    async def test_a_bad_period_is_422(self, client: httpx.AsyncClient) -> None:
        payload = FILING_BODY | {
            "period": {"kind": "quarterly", "fiscal_year": 2025, "end_date": "2025-03-31"}
        }

        response = await client.post("/api/v1/filings", json=payload, headers=WRITE_FILING)

        assert response.status_code == 422

    async def test_an_unknown_filing_is_404(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/filings/fil-missing", headers=READ_ALL)
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    async def test_a_filing_body_is_not_returned(self, client: httpx.AsyncClient) -> None:
        """A 10-K is megabytes; a listing that inlines it is unusable."""
        response = await client.get("/api/v1/filings/fil-1", headers=READ_ALL)

        body = response.json()
        assert response.status_code == 200
        assert "content" not in body
        assert body["content_length"] > 0


class TestStatementsContract:
    async def test_every_figure_is_a_string(self, client: httpx.AsyncClient) -> None:
        """JSON's only numeric type is a double. A figure emitted as a JSON
        number would hand the client a float and undo the Decimal discipline."""
        response = await client.get("/api/v1/entities/ent-acme/statements", headers=READ_ALL)

        line_items = response.json()["income_statement"]["line_items"]
        assert response.status_code == 200
        assert line_items["revenue"] == "412600000"
        assert all(value is None or isinstance(value, str) for value in line_items.values())

    async def test_an_undisclosed_line_is_null_not_zero(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/entities/ent-acme/statements", headers=READ_ALL)

        line_items = response.json()["income_statement"]["line_items"]
        assert line_items["operating_expenses"] is None

    async def test_missing_statements_are_404(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        container.statement_store.present = False

        response = await client.get("/api/v1/entities/ent-acme/statements", headers=READ_ALL)

        assert response.status_code == 404


class TestAnalysisContract:
    async def test_metrics_carry_their_provenance_and_inputs(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.get("/api/v1/entities/ent-acme/analysis", headers=READ_ALL)
        body = response.json()

        assert response.status_code == 200
        net_margin = next(m for m in body["metrics"] if m["name"] == "net_margin")
        assert net_margin["provenance"]["kind"] == "derived"
        assert net_margin["provenance"]["computation"].startswith("atlas.")
        assert net_margin["inputs"]

    async def test_no_computed_metric_names_a_model(self, client: httpx.AsyncClient) -> None:
        """The architectural rule, visible at the API boundary."""
        response = await client.get("/api/v1/entities/ent-acme/analysis", headers=READ_ALL)

        metrics = response.json()["metrics"]
        assert all(metric["provenance"].get("model") is None for metric in metrics)

    async def test_metric_values_are_strings(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/entities/ent-acme/analysis", headers=READ_ALL)

        assert all(isinstance(metric["value"], str) for metric in response.json()["metrics"])

    async def test_what_the_filing_did_not_support_is_returned_separately(
        self, client: httpx.AsyncClient
    ) -> None:
        """Not disclosed is an answer. Burying it makes a partial analysis look
        complete."""
        response = await client.get("/api/v1/entities/ent-acme/analysis", headers=READ_ALL)
        body = response.json()

        assert "unavailable" in body
        assert all(metric["unavailable_reason"] for metric in body["unavailable"])


class TestValuationContract:
    async def test_a_valuation_returns_its_assumptions(self, client: httpx.AsyncClient) -> None:
        """A number nobody can argue with is a number nobody should act on."""
        response = await client.post("/api/v1/valuation/dcf", json=VALUATION_BODY, headers=READ_ALL)
        body = response.json()

        assert response.status_code == 200
        assert body["assumptions"]["discount_rate"] == "0.10"
        assert body["provenance"]["kind"] == "estimate"
        assert body["provenance"]["assumptions"]

    async def test_a_percentage_point_rate_is_refused(self, client: httpx.AsyncClient) -> None:
        """ "10" meaning ten percent is the factor-of-100 error, caught at the edge."""
        response = await client.post(
            "/api/v1/valuation/dcf",
            json=VALUATION_BODY | {"discount_rate": "10"},
            headers=READ_ALL,
        )
        assert response.status_code == 422

    async def test_a_float_rate_is_422_not_500(self, client: httpx.AsyncClient) -> None:
        """A JSON number is already a float by the time Pydantic sees it."""
        response = await client.post(
            "/api/v1/valuation/dcf",
            json=VALUATION_BODY | {"discount_rate": 0.1},
            headers=READ_ALL,
        )
        assert response.status_code == 422

    async def test_a_discount_rate_below_terminal_growth_is_422(
        self, client: httpx.AsyncClient
    ) -> None:
        """A terminal value computed from a negative spread diverges."""
        response = await client.post(
            "/api/v1/valuation/dcf",
            json=VALUATION_BODY | {"discount_rate": "0.02", "terminal_growth": "0.05"},
            headers=READ_ALL,
        )
        assert response.status_code == 422

    async def test_valuation_figures_are_strings(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/v1/valuation/dcf", json=VALUATION_BODY, headers=READ_ALL)
        body = response.json()

        assert isinstance(body["enterprise_value"], str)
        assert Decimal(body["enterprise_value"]) > 0

    async def test_an_unvaluable_filing_is_422_not_an_invented_number(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        container.research_pipeline.valuation = "unavailable"

        response = await client.post("/api/v1/valuation/dcf", json=VALUATION_BODY, headers=READ_ALL)

        assert response.status_code == 422
        assert "do not support a valuation" in response.json()["message"]


class TestResearchContract:
    async def test_a_grounded_report_is_publishable(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/research/reports", json={"entity_id": "ent-acme"}, headers=WRITE_RESEARCH
        )
        body = response.json()

        assert response.status_code == 201
        assert body["publishable"] is True
        assert body["cited_source_ids"] == ["fil-acme-2025"]

    async def test_an_ungrounded_report_is_returned_labelled_not_suppressed(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        """Suppressing it would hide a systematic model problem behind an empty
        response; a client that checks the flag can never mistake it for analysis."""
        container.research_pipeline.report = sample_report(grounded=False)

        response = await client.post(
            "/api/v1/research/reports", json={"entity_id": "ent-acme"}, headers=WRITE_RESEARCH
        )
        body = response.json()

        assert response.status_code == 201
        assert body["publishable"] is False
        assert body["numerically_grounded"] is False
        assert body["unsupported_figures"] == ["31.2%"]

    async def test_a_report_carries_the_figures_it_was_written_from(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.post(
            "/api/v1/research/reports", json={"entity_id": "ent-acme"}, headers=WRITE_RESEARCH
        )
        body = response.json()

        assert {metric["name"] for metric in body["metrics"]} >= {"net_margin", "gross_margin"}
        assert body["provenance"]["kind"] == "generated"
        assert body["provenance"]["model"] == "fake-model-1"

    async def test_an_entity_with_no_filings_is_404(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        container.research_pipeline.error = NotFoundError("no financial statements are on file")

        response = await client.post(
            "/api/v1/research/reports", json={"entity_id": "ent-nothing"}, headers=WRITE_RESEARCH
        )

        assert response.status_code == 404

    async def test_listing_returns_ungrounded_reports_by_default(
        self, client: httpx.AsyncClient, container: StubContainer
    ) -> None:
        """An ungrounded report is evidence about the model. Hiding it by default
        would make that evidence hard to find."""
        container.report_repository.stored = {
            "rpt-1": sample_report(),
            "rpt-2": sample_report(grounded=False),
        }

        every = await client.get("/api/v1/research/entities/ent-acme/reports", headers=READ_ALL)
        filtered = await client.get(
            "/api/v1/research/entities/ent-acme/reports?publishable_only=true", headers=READ_ALL
        )

        assert len(every.json()) == 2
        assert len(filtered.json()) == 1


class TestErrorEnvelope:
    async def test_a_correlation_id_is_echoed(self, client: httpx.AsyncClient) -> None:
        response = await client.get(
            "/health/live", headers={CORRELATION_HEADER: "corr-from-caller"}
        )
        assert response.headers[CORRELATION_HEADER] == "corr-from-caller"

    async def test_an_unhandled_error_leaks_nothing_but_keeps_the_id(
        self, container: StubContainer
    ) -> None:
        container.research_pipeline.error = RuntimeError("connection string: postgres://secret")
        app = create_app(container=container, configure_observability=False)  # type: ignore[arg-type]
        app.dependency_overrides[get_session] = lambda: None

        # Starlette generates the 500 body from the handler and then re-raises so
        # the server can log it. Under uvicorn that is what puts the traceback in
        # the logs; here it has to be suppressed to inspect what the client got.
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://atlas.test"
            ) as isolated:
                response = await isolated.post(
                    "/api/v1/research/reports",
                    json={"entity_id": "ent-acme"},
                    headers=WRITE_RESEARCH | {CORRELATION_HEADER: "corr-trace-me"},
                )

        body = response.json()
        assert response.status_code == 500
        assert "secret" not in response.text
        assert body["message"] == "an unexpected error occurred"
        assert body["correlation_id"] == "corr-trace-me"

    async def test_a_validation_error_names_the_offending_field(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.post(
            "/api/v1/filings", json=FILING_BODY | {"entity_id": ""}, headers=WRITE_FILING
        )
        body = response.json()

        assert response.status_code == 422
        assert body["code"] == "validation_error"
        assert body["details"]["errors"]


class TestOpenAPI:
    async def test_every_route_declares_authorization(self, client: httpx.AsyncClient) -> None:
        """An unprotected endpoint should be visible in the routing table rather
        than hiding in a function body — so this asserts on the table."""
        from atlas.api.app import create_app as build

        app = build(configure_observability=False, container=StubContainer())  # type: ignore[arg-type]
        unprotected = []
        for route in app.routes:
            path = getattr(route, "path", "")
            if not path.startswith("/api/"):
                continue
            names = [d.dependency.__name__ for d in getattr(route, "dependencies", [])]
            if "dependency" not in names:
                unprotected.append(path)

        assert unprotected == []

    async def test_filing_type_is_an_enum_in_the_schema(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/openapi.json")
        schema = response.json()

        assert response.status_code == 200
        assert "10-K" in schema["components"]["schemas"]["FilingType"]["enum"]
