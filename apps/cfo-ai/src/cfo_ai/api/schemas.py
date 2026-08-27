"""Request and response contracts for the CFO.ai API.

Every figure crosses the wire as a **string**, in both directions. JSON has one
numeric type and it is a double, so a response emitting ``123456.7891`` as a
number hands the client a float and undoes the Decimal discipline at the last
step; a posted JSON number is already a float by the time Pydantic sees it, and
is refused rather than rounded.

Every variance also crosses the wire with its **direction spelled out** — the
sign, the verdict, and a sentence that states both. A client that has to infer
"below plan is good here and bad there" from the sign of a number will
eventually infer it wrong, and that is the failure this whole product is
arranged around.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import Field, field_validator, model_validator

from cfo_ai.analysis.compare import CentreRollup
from cfo_ai.analysis.variance import Favourability, Variance, VarianceReport
from cfo_ai.domain.accounts import Account, AccountType
from cfo_ai.domain.organisation import CostCentre, CostCentreTree
from cfo_ai.domain.plan import Plan, PlanKind, PlanLine
from cfo_ai.pipeline import PlanStored
from cfo_ai.research.commentary import Commentary
from fie_finance.metrics import Metric
from fie_finance.money import Money, to_decimal
from fie_finance.periods import FiscalPeriod, PeriodKind
from fie_schemas.base import FIEModel, FrozenModel
from fie_schemas.provenance import Provenance, SourceReference, SourceType


def _decimal_string(value: str) -> Decimal:
    """Parse a client-supplied figure, refusing the lossy forms.

    Raises:
        ValueError: on a float, a boolean, or anything not a finite number.
            Pydantic turns this into a 422, so a client posting a JSON number is
            told its request was malformed rather than receiving a rounded
            answer.
    """
    try:
        return to_decimal(value)
    except TypeError as error:
        raise ValueError(str(error)) from error


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------


class PeriodRequest(FIEModel):
    """A fiscal period."""

    kind: PeriodKind
    fiscal_year: int = Field(ge=1900, le=2200)
    fiscal_quarter: int | None = Field(default=None, ge=1, le=4)
    start_date: date | None = None
    end_date: date

    @model_validator(mode="after")
    def _must_be_a_valid_period(self) -> PeriodRequest:
        """Delegate to the domain rather than restate what makes a period valid.

        Restating it would let the two drift; delegating means a request the
        domain would reject is a 422 here instead of an exception raised deeper
        in a handler and served as a 500.
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


def _period_view(period: FiscalPeriod) -> PeriodView:
    return PeriodView(
        kind=period.kind,
        fiscal_year=period.fiscal_year,
        fiscal_quarter=period.fiscal_quarter,
        start_date=period.start_date,
        end_date=period.end_date,
        label=period.label,
    )


# ---------------------------------------------------------------------------
# Chart of accounts and organisation
# ---------------------------------------------------------------------------


class AccountRequest(FIEModel):
    """One account. ``type`` is the field that decides variance direction."""

    code: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=200)
    type: AccountType
    parent_code: str | None = Field(default=None, max_length=32)
    aliases: list[str] = Field(default_factory=list)

    def to_account(self) -> Account:
        return Account(
            code=self.code,
            name=self.name,
            type=self.type,
            parent_code=self.parent_code,
            aliases=self.aliases,
        )


class PublishChartRequest(FIEModel):
    """Replace a company's chart of accounts."""

    entity_id: str = Field(min_length=1, max_length=64)
    accounts: list[AccountRequest] = Field(min_length=1)

    @model_validator(mode="after")
    def _accounts_must_be_valid(self) -> PublishChartRequest:
        """Delegate to the domain, so a malformed code is a 422 rather than an
        exception raised deeper in the handler and served as a 500."""
        for account in self.accounts:
            account.to_account()
        return self


class AccountView(FrozenModel):
    code: str
    name: str
    type: AccountType
    parent_code: str | None = None
    aliases: list[str] = Field(default_factory=list)
    #: Whether over-plan and under-plan carry a value judgement for this
    #: account at all. False for headcount.
    has_direction: bool
    #: Whether a larger number is the better outcome. Meaningless when
    #: ``has_direction`` is false.
    higher_is_better: bool

    @classmethod
    def from_account(cls, account: Account) -> AccountView:
        return cls(
            code=account.code,
            name=account.name,
            type=account.type,
            parent_code=account.parent_code,
            aliases=list(account.aliases),
            has_direction=account.type.has_direction,
            higher_is_better=account.type.is_inflow,
        )


class CostCentreRequest(FIEModel):
    id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    parent_id: str | None = Field(default=None, max_length=64)
    owner: str | None = Field(default=None, max_length=200)


class PublishOrganisationRequest(FIEModel):
    """Replace a company's cost centre hierarchy."""

    entity_id: str = Field(min_length=1, max_length=64)
    centres: list[CostCentreRequest] = Field(min_length=1)

    @model_validator(mode="after")
    def _must_be_a_tree(self) -> PublishOrganisationRequest:
        """Delegate to the domain, so a cycle is a 422 rather than a 500."""
        self.to_tree()
        return self

    def to_tree(self) -> CostCentreTree:
        return CostCentreTree(
            centres=[
                CostCentre(
                    id=centre.id,
                    name=centre.name,
                    parent_id=centre.parent_id,
                    owner=centre.owner,
                )
                for centre in self.centres
            ]
        )


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


class PlanLineRequest(FIEModel):
    """One planned or booked amount. ``amount`` is a string, never a number."""

    account_code: str = Field(min_length=1, max_length=32)
    cost_centre_id: str = Field(min_length=1, max_length=64)
    period: PeriodRequest
    amount: str
    memo: str | None = Field(default=None, max_length=500)

    @field_validator("amount")
    @classmethod
    def _parse_amount(cls, value: str) -> str:
        _decimal_string(value)
        return value


class SubmitPlanRequest(FIEModel):
    """Submit a budget, reforecast, or set of actuals."""

    entity_id: str = Field(min_length=1, max_length=64)
    kind: PlanKind
    period: PeriodRequest
    currency: str = Field(default="USD", min_length=3, max_length=3)
    version: str = Field(default="v1", min_length=1, max_length=64)
    lines: list[PlanLineRequest] = Field(min_length=1)
    #: Where the figures came from — the board pack, a ledger export. Required,
    #: because a budget nobody can trace is a figure with no owner.
    source_id: str = Field(min_length=1, max_length=64)
    source_title: str | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def _must_be_a_valid_plan(self) -> SubmitPlanRequest:
        """Build the domain object here so its rules answer at the edge.

        Two lines for one account, centre, and period is the case that matters:
        the domain refuses it, and without this the refusal surfaces from inside
        the handler as a 500 — telling a client its own duplicate row was our
        outage. Constructing twice costs nothing next to the database write that
        follows.
        """
        self.to_plan()
        return self

    def to_plan(self) -> Plan:
        period = self.period.to_period()
        return Plan(
            entity_id=self.entity_id,
            kind=self.kind,
            period=period,
            currency=self.currency,
            version=self.version,
            lines=[
                PlanLine(
                    account_code=item.account_code,
                    cost_centre_id=item.cost_centre_id,
                    period=item.period.to_period(),
                    amount=Money.of(item.amount, self.currency),
                    memo=item.memo,
                )
                for item in self.lines
            ],
            provenance=Provenance.fact(
                SourceReference(
                    source_id=self.source_id,
                    source_type=SourceType.INTERNAL_DOCUMENT,
                    title=self.source_title,
                )
            ),
        )


class SubmitPlanResponse(FrozenModel):
    """What storing a plan produced."""

    plan_id: str
    entity_id: str
    line_count: int
    total: str
    #: Cost centres the hierarchy does not contain. Their spend rolls up
    #: nowhere, so it is returned rather than silently absorbed.
    unassigned_cost_centres: list[str] = Field(default_factory=list)

    @classmethod
    def from_stored(cls, stored: PlanStored) -> SubmitPlanResponse:
        return cls(
            plan_id=stored.plan_id,
            entity_id=stored.entity_id,
            line_count=stored.line_count,
            total=stored.total,
            unassigned_cost_centres=list(stored.unassigned_cost_centres),
        )


class PlanLineView(FrozenModel):
    account_code: str
    cost_centre_id: str
    period: PeriodView
    amount: str
    memo: str | None = None


class PlanView(FrozenModel):
    id: str
    entity_id: str
    kind: PlanKind
    period: PeriodView
    currency: str
    version: str
    total: str
    lines: list[PlanLineView] = Field(default_factory=list)
    provenance: Provenance

    @classmethod
    def from_plan(cls, plan: Plan) -> PlanView:
        return cls(
            id=plan.id,
            entity_id=plan.entity_id,
            kind=plan.kind,
            period=_period_view(plan.period),
            currency=plan.currency,
            version=plan.version,
            total=str(plan.total().amount),
            lines=[
                PlanLineView(
                    account_code=line.account_code,
                    cost_centre_id=line.cost_centre_id,
                    period=_period_view(line.period),
                    amount=str(line.amount.amount),
                    memo=line.memo,
                )
                for line in plan.lines
            ],
            provenance=plan.provenance,
        )


# ---------------------------------------------------------------------------
# Variance
# ---------------------------------------------------------------------------


class VarianceView(FrozenModel):
    """One line's variance, with the direction stated rather than implied."""

    account_code: str
    account_name: str
    account_type: AccountType
    cost_centre_id: str
    budget: str
    actual: str
    #: ``actual - budget``. Positive means higher than plan, which is good or
    #: bad depending on the account — see ``favourability``.
    amount: str
    #: ``None`` when the plan was zero. Never infinity.
    percent: str | None = None
    favourability: Favourability
    is_material: bool
    #: A sentence a human can read, with the direction spelled out.
    description: str

    @classmethod
    def from_variance(cls, variance: Variance) -> VarianceView:
        return cls(
            account_code=variance.account_code,
            account_name=variance.account_name,
            account_type=variance.account_type,
            cost_centre_id=variance.cost_centre_id,
            budget=str(variance.budget.amount),
            actual=str(variance.actual.amount),
            amount=str(variance.amount.amount),
            percent=str(variance.percent) if variance.percent is not None else None,
            favourability=variance.favourability,
            is_material=variance.is_material,
            description=variance.describe(),
        )


class RollupView(FrozenModel):
    cost_centre_id: str
    cost_centre_name: str
    depth: int
    budget: str
    actual: str
    variance: str
    child_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_rollup(cls, rollup: CentreRollup) -> RollupView:
        return cls(
            cost_centre_id=rollup.cost_centre_id,
            cost_centre_name=rollup.cost_centre_name,
            depth=rollup.depth,
            budget=str(rollup.budget.amount),
            actual=str(rollup.actual.amount),
            variance=str(rollup.variance.amount),
            child_ids=list(rollup.child_ids),
        )


class VarianceResponse(FrozenModel):
    """A period's variances.

    Favourable, unfavourable, and not-assessed are returned as separate lists
    rather than one a client has to bucket. A client doing its own bucketing
    would have to re-derive the direction rule, which is the one thing this
    service exists to decide once.
    """

    entity_id: str
    period: PeriodView
    currency: str
    net_operating_variance: str
    variances: list[VarianceView] = Field(default_factory=list)
    material: list[VarianceView] = Field(default_factory=list)
    unfavourable: list[VarianceView] = Field(default_factory=list)
    favourable: list[VarianceView] = Field(default_factory=list)
    #: Lines whose direction the service declines to judge, such as headcount.
    not_assessed: list[VarianceView] = Field(default_factory=list)
    #: Spend booked with no budget line, and budget lines with nothing booked.
    unplanned_spend_keys: list[str] = Field(default_factory=list)
    unspent_budget_keys: list[str] = Field(default_factory=list)
    rollups: list[RollupView] = Field(default_factory=list)

    @classmethod
    def from_report(
        cls, report: VarianceReport, rollups: list[CentreRollup] | None = None
    ) -> VarianceResponse:
        render = VarianceView.from_variance
        return cls(
            entity_id=report.entity_id,
            period=_period_view(report.period),
            currency=report.currency,
            net_operating_variance=str(report.net_operating_variance().amount),
            variances=[render(item) for item in report.variances],
            material=[render(item) for item in report.material],
            unfavourable=[render(item) for item in report.unfavourable],
            favourable=[render(item) for item in report.favourable],
            not_assessed=[render(item) for item in report.not_assessed],
            unplanned_spend_keys=list(report.unmatched_actual_keys),
            unspent_budget_keys=list(report.unmatched_budget_keys),
            rollups=[RollupView.from_rollup(item) for item in (rollups or [])],
        )


# ---------------------------------------------------------------------------
# Commentary
# ---------------------------------------------------------------------------


class MetricView(FrozenModel):
    name: str
    value: str
    unit: str
    currency: str | None = None
    rendered: str
    inputs: dict[str, str] = Field(default_factory=dict)
    provenance: Provenance
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


class GenerateCommentaryRequest(FIEModel):
    """Ask for management commentary on a period's variances."""

    entity_id: str = Field(min_length=1, max_length=64)
    company_name: str | None = Field(default=None, max_length=512)
    period: PeriodRequest
    budget_version: str | None = Field(default=None, max_length=64)


class CommentarySectionView(FrozenModel):
    heading: str
    body: str


class CommentaryResponse(FrozenModel):
    """Commentary and both verification verdicts.

    ``publishable`` requires both. A commentary that quotes a real number and
    draws the opposite conclusion from it is not a lesser failure than one that
    invents a number — it is a more persuasive one — so it is returned labelled
    rather than hidden.
    """

    commentary_id: str
    entity_id: str
    period_label: str
    summary: str
    sections: list[CommentarySectionView] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    metrics: list[MetricView] = Field(default_factory=list)
    publishable: bool
    numerically_grounded: bool
    unsupported_figures: list[str] = Field(default_factory=list)
    directionally_grounded: bool
    #: Explanations of any sentence that reversed the meaning of a line.
    direction_contradictions: list[str] = Field(default_factory=list)
    regeneration_count: int = 0
    provenance: Provenance

    @classmethod
    def from_commentary(cls, commentary_id: str, commentary: Commentary) -> CommentaryResponse:
        return cls(
            commentary_id=commentary_id,
            entity_id=commentary.entity_id,
            period_label=commentary.period_label,
            summary=commentary.summary,
            sections=[
                CommentarySectionView(heading=section.heading, body=section.body)
                for section in commentary.sections
            ],
            open_questions=list(commentary.open_questions),
            metrics=[MetricView.from_metric(metric) for metric in commentary.metrics],
            publishable=commentary.is_publishable,
            numerically_grounded=commentary.numerically_grounded,
            unsupported_figures=list(commentary.unsupported_figures),
            directionally_grounded=commentary.directionally_grounded,
            direction_contradictions=list(commentary.direction_contradictions),
            regeneration_count=commentary.regeneration_count,
            provenance=commentary.provenance,
        )


__all__ = [
    "AccountRequest",
    "AccountView",
    "CommentaryResponse",
    "CommentarySectionView",
    "CostCentreRequest",
    "GenerateCommentaryRequest",
    "MetricView",
    "PeriodRequest",
    "PeriodView",
    "PlanLineRequest",
    "PlanLineView",
    "PlanView",
    "PublishChartRequest",
    "PublishOrganisationRequest",
    "RollupView",
    "SubmitPlanRequest",
    "SubmitPlanResponse",
    "VarianceResponse",
    "VarianceView",
]
