"""Request and response contracts for the Atlas API.

Separate from the domain models on purpose: a domain ``IncomeStatement``
carries provenance objects and validators that no HTTP client needs, and
exposing it directly would make every internal refactor a breaking API change.

One rule runs through all of it. **Every figure crosses the wire as a string.**
JSON has one numeric type and it is a double, so a response that emitted
``412600000.5`` as a JSON number would hand the client a float and quietly undo
the Decimal discipline the whole service is built on. Requests are strings for
the mirror-image reason: a posted JSON number is already a float by the time
Pydantic sees it, and Atlas rejects it rather than rounding it.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from pydantic import Field, field_validator, model_validator

from atlas.analysis.dcf import DCFValuation
from atlas.domain.statements import FinancialStatements, StatementBase
from atlas.filings.models import Filing, FilingType
from atlas.filings.pipeline import IngestionResult
from atlas.research.reports import ResearchReport
from fie_finance.metrics import AnalysisResult, Metric
from fie_finance.money import Money, Shares, to_decimal
from fie_finance.periods import FiscalPeriod, PeriodKind
from fie_schemas.base import FIEModel, FrozenModel
from fie_schemas.provenance import Provenance


def _decimal_string(value: str) -> Decimal:
    """Parse a client-supplied figure, refusing the lossy forms.

    Raises:
        ValueError: on a float, a boolean, or anything not a finite number.
            Pydantic turns this into a 422, so a client that posts
            ``{"amount": 1234.56}`` is told its request was malformed rather
            than receiving a silently rounded answer.
    """
    try:
        return to_decimal(value)
    except TypeError as error:
        raise ValueError(str(error)) from error


# ---------------------------------------------------------------------------
# Filings
# ---------------------------------------------------------------------------


class PeriodRequest(FIEModel):
    """The fiscal period a filing covers."""

    kind: PeriodKind
    fiscal_year: int = Field(ge=1900, le=2200)
    fiscal_quarter: int | None = Field(default=None, ge=1, le=4)
    start_date: date | None = None
    end_date: date

    @model_validator(mode="after")
    def _must_be_a_valid_period(self) -> PeriodRequest:
        """Enforce the domain's own rules at the edge.

        The check delegates to :class:`FiscalPeriod` rather than restating what
        makes a period valid. Restating it would let the two drift; delegating
        means a request that the domain would reject is a 422 here, instead of
        an exception raised deeper in the handler and served as a 500.
        """
        self.to_period()
        return self

    def to_period(self) -> FiscalPeriod:
        return FiscalPeriod(
            kind=self.kind,
            fiscal_year=self.fiscal_year,
            fiscal_quarter=self.fiscal_quarter,
            start_date=self.start_date,
            end_date=self.end_date,
        )


class PeriodView(FrozenModel):
    kind: PeriodKind
    fiscal_year: int
    fiscal_quarter: int | None = None
    start_date: date | None = None
    end_date: date
    label: str


class IngestFilingRequest(FIEModel):
    """Submit a filing for storage and extraction."""

    entity_id: str = Field(min_length=1, max_length=64)
    type: FilingType
    period: PeriodRequest
    title: str = Field(min_length=1, max_length=1024)
    content: str = Field(min_length=1)
    accession_number: str | None = Field(default=None, max_length=64)
    filed_at: date | None = None
    uri: str | None = Field(default=None, max_length=2048)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_filing(self) -> Filing:
        return Filing(
            entity_id=self.entity_id,
            type=self.type,
            period=self.period.to_period(),
            title=self.title,
            content=self.content,
            accession_number=self.accession_number,
            filed_at=self.filed_at,
            uri=self.uri,
            metadata=self.metadata,
        )


class IngestFilingResponse(FrozenModel):
    """What ingestion produced.

    ``rejected`` is returned even when statements were extracted successfully.
    A set that survived with three refused line items is a different object from
    a clean one, and a client that cannot see the difference will treat them the
    same.
    """

    filing_id: str
    entity_id: str
    period_label: str
    newly_ingested: bool
    statement_set_id: str | None = None
    has_income_statement: bool = False
    has_balance_sheet: bool = False
    has_cash_flow_statement: bool = False
    rejected: list[str] = Field(default_factory=list)
    failed_validation: bool = False

    @classmethod
    def from_result(cls, result: IngestionResult) -> IngestFilingResponse:
        return cls(
            filing_id=result.filing_id,
            entity_id=result.entity_id,
            period_label=result.period_label,
            newly_ingested=result.newly_ingested,
            statement_set_id=result.statement_set_id,
            has_income_statement=result.has_income_statement,
            has_balance_sheet=result.has_balance_sheet,
            has_cash_flow_statement=result.has_cash_flow_statement,
            rejected=list(result.rejected),
            failed_validation=result.failed_validation,
        )


class FilingView(FrozenModel):
    """A stored filing, without its body."""

    id: str
    entity_id: str
    type: FilingType
    period: PeriodView
    title: str
    accession_number: str | None = None
    filed_at: date | None = None
    uri: str | None = None
    content_hash: str
    #: Characters of source text held. The body itself is not returned: a 10-K
    #: is megabytes, and a client that wants it should ask for it by id.
    content_length: int

    @classmethod
    def from_filing(cls, filing: Filing) -> FilingView:
        return cls(
            id=filing.id,
            entity_id=filing.entity_id,
            type=filing.type,
            period=_period_view(filing.period),
            title=filing.title,
            accession_number=filing.accession_number,
            filed_at=filing.filed_at,
            uri=filing.uri,
            content_hash=filing.content_hash,
            content_length=len(filing.content),
        )


# ---------------------------------------------------------------------------
# Statements and analysis
# ---------------------------------------------------------------------------


class MoneyView(FrozenModel):
    """A monetary amount, as a string."""

    amount: str
    currency: str


class StatementView(FrozenModel):
    """One statement's line items, all as strings.

    ``None`` means the filing did not disclose the line. It is never rendered as
    zero — a zero reads as a measurement, and "not disclosed" is a different
    claim about the company.
    """

    period: PeriodView
    currency: str
    provenance: Provenance
    line_items: dict[str, str | None]


class StatementSetView(FrozenModel):
    """A period's statements as the API returns them."""

    entity_id: str
    period: PeriodView
    currency: str
    income_statement: StatementView | None = None
    balance_sheet: StatementView | None = None
    cash_flow_statement: StatementView | None = None
    source_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_statements(cls, statements: FinancialStatements) -> StatementSetView:
        return cls(
            entity_id=statements.entity_id,
            period=_period_view(statements.period),
            currency=statements.currency,
            income_statement=_statement_view(statements.income_statement),
            balance_sheet=_statement_view(statements.balance_sheet),
            cash_flow_statement=_statement_view(statements.cash_flow_statement),
            source_ids=statements.source_ids,
        )


def _statement_view(statement: StatementBase | None) -> StatementView | None:
    """Render a statement's monetary lines as strings.

    Derived from the model's own fields rather than an explicit list, so a line
    item added to the domain appears in the API without a second edit — and,
    more importantly, cannot be silently dropped from it.
    """
    if statement is None:
        return None

    line_items: dict[str, str | None] = {}
    for name in type(statement).model_fields:
        if name in ("period", "currency", "provenance"):
            continue
        value = getattr(statement, name)
        if isinstance(value, Money):
            line_items[name] = str(value.amount)
        elif isinstance(value, Shares):
            line_items[name] = str(value.count)
        else:
            # Absent, so reported as absent. Never zero.
            line_items[name] = None

    return StatementView(
        period=_period_view(statement.period),
        currency=statement.currency,
        provenance=statement.provenance,
        line_items=line_items,
    )


class MetricView(FrozenModel):
    """A computed figure and everything needed to reproduce it."""

    name: str
    #: A string, always. See the module docstring.
    value: str
    unit: str
    currency: str | None = None
    rendered: str
    #: The named inputs the value was computed from. A number nobody can
    #: reproduce is a number nobody should act on.
    inputs: dict[str, str] = Field(default_factory=dict)
    provenance: Provenance
    #: Present when the filing did not support the calculation.
    unavailable_reason: str | None = None

    @classmethod
    def from_metric(cls, metric: Metric) -> MetricView:
        return cls(
            name=metric.name,
            value=str(metric.value),
            unit=str(metric.unit),
            currency=metric.currency,
            rendered=metric.rendered(),
            inputs=dict(metric.inputs),
            provenance=metric.provenance,
            unavailable_reason=metric.unavailable_reason,
        )


class AnalysisResponse(FrozenModel):
    """Deterministic metrics for one period.

    Available and unavailable metrics are returned in separate lists rather than
    one list a client has to filter. What a filing does *not* disclose is an
    answer, and burying it makes it easy to present a partial analysis as a
    complete one.
    """

    entity_id: str
    period_label: str
    metrics: list[MetricView] = Field(default_factory=list)
    unavailable: list[MetricView] = Field(default_factory=list)

    @classmethod
    def from_analysis(cls, analysis: AnalysisResult) -> AnalysisResponse:
        return cls(
            entity_id=analysis.entity_id,
            period_label=analysis.period_label,
            metrics=[MetricView.from_metric(metric) for metric in analysis.available],
            unavailable=[MetricView.from_metric(metric) for metric in analysis.unavailable],
        )


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------


class ValuationRequest(FIEModel):
    """Assumptions for a discounted cash flow.

    Required, not defaulted. A discount rate is a judgement about risk, and a
    service that picks one silently turns an arguable estimate into an apparent
    measurement. Rates are decimal fractions as strings: ``"0.10"`` is ten
    percent.
    """

    entity_id: str = Field(min_length=1, max_length=64)
    fiscal_year: int | None = Field(default=None, ge=1900, le=2200)
    discount_rate: str
    terminal_growth: str
    forecast_growth: str
    projection_years: int = Field(default=5, ge=1, le=20)

    @field_validator("discount_rate", "terminal_growth", "forecast_growth")
    @classmethod
    def _parse_rate(cls, value: str) -> str:
        rate = _decimal_string(value)
        if not -1 < rate < 1:
            raise ValueError(
                "a rate is a decimal fraction, not percentage points: pass '0.10' for ten percent"
            )
        return value


class ProjectionView(FrozenModel):
    year: int
    cash_flow: str
    discount_factor: str
    present_value: str


class ValuationResponse(FrozenModel):
    """A DCF result, labelled as an estimate everywhere it appears."""

    entity_id: str
    currency: str
    enterprise_value: str
    equity_value: str | None = None
    value_per_share: str | None = None
    present_value_of_forecast: str
    terminal_value: str
    present_value_of_terminal: str
    projections: list[ProjectionView] = Field(default_factory=list)
    #: The stated inputs, returned with the result so the number is arguable.
    assumptions: dict[str, str] = Field(default_factory=dict)
    #: True when most of the value comes from the terminal value, meaning the
    #: figure rests mainly on the perpetuity assumption rather than the forecast.
    terminal_dominated: bool = False
    provenance: Provenance

    @classmethod
    def from_valuation(cls, valuation: DCFValuation, provenance: Provenance) -> ValuationResponse:
        return cls(
            entity_id=valuation.entity_id,
            currency=valuation.currency,
            enterprise_value=str(valuation.enterprise_value.amount),
            equity_value=(str(valuation.equity_value.amount) if valuation.equity_value else None),
            value_per_share=(
                str(valuation.value_per_share.amount) if valuation.value_per_share else None
            ),
            present_value_of_forecast=str(valuation.present_value_of_forecast.amount),
            terminal_value=str(valuation.terminal_value.amount),
            present_value_of_terminal=str(valuation.present_value_of_terminal.amount),
            projections=[
                ProjectionView(
                    year=projection.year,
                    cash_flow=str(projection.cash_flow.amount),
                    discount_factor=str(projection.discount_factor),
                    present_value=str(projection.present_value.amount),
                )
                for projection in valuation.projections
            ],
            assumptions=dict(valuation.assumptions),
            terminal_dominated=valuation.is_terminal_dominated,
            provenance=provenance,
        )


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------


class ReportRequestBody(FIEModel):
    """Ask for a research note on a company's filed figures."""

    entity_id: str = Field(min_length=1, max_length=64)
    company_name: str | None = Field(default=None, max_length=512)
    fiscal_year: int | None = Field(default=None, ge=1900, le=2200)
    #: Supply to include a DCF in the note. Omitted, the report describes what
    #: was filed and values nothing.
    valuation: ValuationRequest | None = None
    include_graph_context: bool = True


class ReportSectionView(FrozenModel):
    heading: str
    body: str


class ReportResponse(FrozenModel):
    """A research note and the verdict of its verification.

    ``publishable`` is the field that matters. A report quoting a figure Atlas
    never computed is returned rather than hidden — suppressing it would conceal
    a systematic model problem — but it arrives labelled, with the offending
    figures named, so no client can mistake it for analysis.
    """

    report_id: str
    entity_id: str
    period_label: str
    summary: str
    sections: list[ReportSectionView] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    metrics: list[MetricView] = Field(default_factory=list)
    cited_source_ids: list[str] = Field(default_factory=list)
    publishable: bool
    numerically_grounded: bool
    citations_grounded: bool
    unsupported_figures: list[str] = Field(default_factory=list)
    regeneration_count: int = 0
    provenance: Provenance

    @classmethod
    def from_report(cls, report_id: str, report: ResearchReport) -> ReportResponse:
        return cls(
            report_id=report_id,
            entity_id=report.entity_id,
            period_label=report.period_label,
            summary=report.summary,
            sections=[
                ReportSectionView(heading=section.heading, body=section.body)
                for section in report.sections
            ],
            open_questions=list(report.open_questions),
            metrics=[MetricView.from_metric(metric) for metric in report.metrics],
            cited_source_ids=list(report.cited_source_ids),
            publishable=report.is_publishable,
            numerically_grounded=report.numerically_grounded,
            citations_grounded=report.citations_grounded,
            unsupported_figures=list(report.unsupported_figures),
            regeneration_count=report.regeneration_count,
            provenance=report.provenance,
        )


def _period_view(period: Any) -> PeriodView:
    return PeriodView(
        kind=period.kind,
        fiscal_year=period.fiscal_year,
        fiscal_quarter=period.fiscal_quarter,
        start_date=period.start_date,
        end_date=period.end_date,
        label=period.label,
    )


__all__ = [
    "AnalysisResponse",
    "FilingView",
    "IngestFilingRequest",
    "IngestFilingResponse",
    "MetricView",
    "MoneyView",
    "PeriodRequest",
    "PeriodView",
    "ProjectionView",
    "ReportRequestBody",
    "ReportResponse",
    "ReportSectionView",
    "StatementSetView",
    "StatementView",
    "ValuationRequest",
    "ValuationResponse",
]
