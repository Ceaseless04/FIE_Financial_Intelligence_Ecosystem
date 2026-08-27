"""Domain events CFO.ai publishes.

CFO.ai is where a company's own numbers live, so its events carry things other
products act on: Sentinel watches for spend that nobody budgeted, FinOps
reconciles infrastructure cost against the plan, and Venture compares a
portfolio company's burn against its stated runway.

``cfo.commentary.published`` and ``cfo.commentary.withheld`` are separate types
for the same reason Atlas splits its report events, and one more besides.
Commentary here can fail in two distinct ways — an invented figure, or a correct
figure with the direction reversed — and the second is the more persuasive
failure. A consumer must opt into that case deliberately rather than miss a
boolean on an event it already trusted.

Payloads are typed here rather than assembled as loose dicts at the call site,
so a field rename shows up as a failing build in the producer instead of a
missing key in a consumer.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from fie_events.schemas import DomainEvent
from fie_schemas.base import FrozenModel

SOURCE_APP = "cfo-ai"

#: A budget, reforecast, or actuals set was stored.
PLAN_STORED = "cfo.plan.stored"
#: A chart of accounts was published, replacing whatever was there.
CHART_UPDATED = "cfo.chart.updated"
#: A variance report was produced for a period.
VARIANCE_COMPUTED = "cfo.variance.computed"
#: Spend was booked against an account and centre with no budget line.
UNPLANNED_SPEND_DETECTED = "cfo.variance.unplanned_spend"
#: A forecast was produced. Always an estimate; assumptions travel with it.
FORECAST_PRODUCED = "cfo.forecast.produced"
#: Runway was computed, or explicitly found not to apply.
RUNWAY_ASSESSED = "cfo.cash.runway_assessed"
#: A scenario was modelled against a baseline.
SCENARIO_MODELLED = "cfo.scenario.modelled"
#: Commentary passed both grounding checks and may be shown as analysis.
COMMENTARY_PUBLISHED = "cfo.commentary.published"
#: Commentary failed a grounding check. Recorded, never publishable.
COMMENTARY_WITHHELD = "cfo.commentary.withheld"

#: Every type this app publishes. Consumers subscribe by prefix, but an explicit
#: list keeps the contract greppable from the other five apps.
PUBLISHED_EVENT_TYPES: tuple[str, ...] = (
    PLAN_STORED,
    CHART_UPDATED,
    VARIANCE_COMPUTED,
    UNPLANNED_SPEND_DETECTED,
    FORECAST_PRODUCED,
    RUNWAY_ASSESSED,
    SCENARIO_MODELLED,
    COMMENTARY_PUBLISHED,
    COMMENTARY_WITHHELD,
)


class PlanStoredPayload(FrozenModel):
    """A plan is on file and can be compared against."""

    plan_id: str
    entity_id: str
    kind: str
    period_label: str
    version: str
    currency: str
    line_count: int = Field(ge=0)
    total: str
    #: Cost centres the plan references that the stored hierarchy does not
    #: contain. Their spend is invisible to every roll-up until the org chart
    #: catches up, so it is announced rather than silently absorbed.
    unassigned_cost_centres: list[str] = Field(default_factory=list)


class ChartUpdatedPayload(FrozenModel):
    """A chart of accounts was replaced.

    Carries the account types because they are what decide variance direction
    downstream — a consumer caching the chart needs to know they changed.
    """

    entity_id: str
    account_count: int = Field(ge=0)
    account_types: dict[str, str] = Field(default_factory=dict)


class VarianceComputedPayload(FrozenModel):
    """A period's variances, summarised."""

    entity_id: str
    period_label: str
    currency: str
    net_operating_variance: str
    material_count: int = Field(ge=0)
    unfavourable_count: int = Field(ge=0)
    favourable_count: int = Field(ge=0)
    #: Lines whose direction the system declined to judge. Reported because a
    #: consumer counting only the two verdicts would silently lose them.
    not_assessed_count: int = Field(ge=0)
    #: The worst lines by absolute money, as account codes.
    largest_variance_accounts: list[str] = Field(default_factory=list)


class UnplannedSpendPayload(FrozenModel):
    """Money booked where no budget line exists.

    Its own event type rather than a field on the variance summary: unplanned
    spend is what a controller chases, and burying it inside a totals payload
    means it arrives only for consumers that already read the whole report.
    """

    entity_id: str
    period_label: str
    #: ``account/centre/period`` keys with actuals and no budget.
    keys: list[str] = Field(default_factory=list)


class ForecastProducedPayload(FrozenModel):
    """A projection. An estimate, and labelled as one everywhere it travels."""

    entity_id: str
    account_code: str
    method: str
    periods: int = Field(ge=1)
    total: str
    currency: str
    #: The stated inputs. A projection without them is not arguable.
    assumptions: dict[str, str] = Field(default_factory=dict)


class RunwayAssessedPayload(FrozenModel):
    """Cash runway, or an explicit statement that the question does not apply."""

    entity_id: str
    period_label: str
    posture: str
    #: ``None`` when the company is generating cash or at break-even. Never a
    #: sentinel, and never a very large number standing in for "not applicable".
    months: str | None = None
    net_burn_per_month: str
    cash_on_hand: str
    currency: str
    unavailable_reason: str | None = None


class ScenarioModelledPayload(FrozenModel):
    """A scenario's effect on a baseline."""

    entity_id: str
    scenario_name: str
    period_label: str
    baseline_total: str
    scenario_total: str
    change: str
    currency: str
    lines_changed: int = Field(ge=0)
    assumptions: dict[str, str] = Field(default_factory=dict)


class CommentaryPayload(FrozenModel):
    """A commentary's identity and both verification verdicts."""

    commentary_id: str
    entity_id: str
    period_label: str
    numerically_grounded: bool
    directionally_grounded: bool
    #: Figures the model introduced that CFO.ai did not compute.
    unsupported_figures: list[str] = Field(default_factory=list)
    #: Sentences that reversed the meaning of a line, explained. Empty on a
    #: published commentary by construction.
    direction_contradictions: list[str] = Field(default_factory=list)
    regeneration_count: int = Field(default=0, ge=0)


def build_event(
    event_type: str,
    payload: FrozenModel,
    *,
    correlation_id: str | None = None,
    causation_id: str | None = None,
    tenant_id: str | None = None,
) -> DomainEvent:
    """Wrap a typed payload in the shared envelope."""
    body: dict[str, Any] = payload.model_dump(mode="json")
    return DomainEvent(
        event_type=event_type,
        source_app=SOURCE_APP,
        correlation_id=correlation_id,
        causation_id=causation_id,
        tenant_id=tenant_id,
        payload=body,
    )


__all__ = [
    "CHART_UPDATED",
    "COMMENTARY_PUBLISHED",
    "COMMENTARY_WITHHELD",
    "FORECAST_PRODUCED",
    "PLAN_STORED",
    "PUBLISHED_EVENT_TYPES",
    "RUNWAY_ASSESSED",
    "SCENARIO_MODELLED",
    "SOURCE_APP",
    "UNPLANNED_SPEND_DETECTED",
    "VARIANCE_COMPUTED",
    "ChartUpdatedPayload",
    "CommentaryPayload",
    "ForecastProducedPayload",
    "PlanStoredPayload",
    "RunwayAssessedPayload",
    "ScenarioModelledPayload",
    "UnplannedSpendPayload",
    "VarianceComputedPayload",
    "build_event",
]
