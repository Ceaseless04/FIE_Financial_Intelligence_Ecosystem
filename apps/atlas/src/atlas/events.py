"""Domain events Atlas publishes.

Atlas is the ecosystem's analytical layer: CFO.ai compares a company's own
numbers against filed peers, Venture reuses the valuation machinery, and
Sentinel watches for the risk language a filing introduces. Those products act
on Atlas's output, which makes one distinction an integration concern rather
than an internal detail.

``atlas.report.published`` and ``atlas.report.withheld`` are separate event
types for exactly that reason. A report whose prose quotes a figure Atlas never
computed is not a slightly worse report — it is one no downstream product may
treat as analysis. Making that a different event type, rather than a boolean
field on one, means a consumer has to opt into the ungrounded case deliberately
instead of missing a flag.

Payloads are typed here rather than assembled as loose dicts at the call site,
so a field rename shows up as a failing build in the producer instead of a
missing key in a consumer.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from fie_events.schemas import DomainEvent
from fie_schemas.base import FrozenModel

SOURCE_APP = "atlas"

#: A filing was stored and is now citable.
FILING_INGESTED = "atlas.filing.ingested"
#: A filing arrived that had already been ingested; no extraction was run.
FILING_DUPLICATE = "atlas.filing.duplicate"
#: Statements were transcribed and passed their accounting identities.
STATEMENTS_EXTRACTED = "atlas.statements.extracted"
#: Transcription produced figures that failed validation. The filing is on
#: record, the statements are not.
STATEMENTS_REJECTED = "atlas.statements.rejected"
#: Deterministic ratios were computed for a period.
ANALYSIS_COMPLETED = "atlas.analysis.completed"
#: A DCF was produced. Always an estimate; the assumptions travel with it.
VALUATION_PRODUCED = "atlas.valuation.produced"
#: A research report passed both grounding checks and may be shown as analysis.
REPORT_PUBLISHED = "atlas.report.published"
#: A research report failed a grounding check. Recorded, never publishable.
REPORT_WITHHELD = "atlas.report.withheld"

#: Every type this app publishes. Consumers subscribe by prefix, but an explicit
#: list keeps the contract greppable from the other five apps.
PUBLISHED_EVENT_TYPES: tuple[str, ...] = (
    FILING_INGESTED,
    FILING_DUPLICATE,
    STATEMENTS_EXTRACTED,
    STATEMENTS_REJECTED,
    ANALYSIS_COMPLETED,
    VALUATION_PRODUCED,
    REPORT_PUBLISHED,
    REPORT_WITHHELD,
)


class FilingIngestedPayload(FrozenModel):
    """A filing is stored and citable."""

    filing_id: str
    entity_id: str
    filing_type: str
    period_label: str
    title: str
    content_hash: str
    uri: str | None = None


class FilingDuplicatePayload(FrozenModel):
    """A redelivered filing that matched an existing content hash."""

    filing_id: str
    existing_filing_id: str
    entity_id: str
    content_hash: str


class StatementsExtractedPayload(FrozenModel):
    """Validated figures are on file for a period."""

    statement_set_id: str
    filing_id: str
    entity_id: str
    period_label: str
    currency: str
    has_income_statement: bool
    has_balance_sheet: bool
    has_cash_flow_statement: bool
    #: What the extractor refused while still producing a usable set. A
    #: consumer that treats a set with rejections as complete is drawing a
    #: conclusion the data does not support.
    rejected: list[str] = Field(default_factory=list)


class StatementsRejectedPayload(FrozenModel):
    """Transcription failed its arithmetic checks.

    Published rather than logged: a filing whose balance sheet will not balance
    is a fact about that filing, and repeated rejections for one filer are a
    signal that Sentinel and CFO.ai both want.
    """

    filing_id: str
    entity_id: str
    period_label: str
    reasons: list[str] = Field(default_factory=list)
    #: True when figures were read but contradicted each other, as opposed to
    #: nothing having been found at all.
    failed_validation: bool = False


class AnalysisCompletedPayload(FrozenModel):
    """Deterministic metrics for a period."""

    entity_id: str
    period_label: str
    #: Metrics actually computed, by name.
    metric_names: list[str] = Field(default_factory=list)
    #: Metrics the filing did not support, by name. Carried because "not
    #: disclosed" is information, and a consumer that only sees the available
    #: set cannot tell a thin filing from a complete one.
    unavailable_names: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class ValuationProducedPayload(FrozenModel):
    """A DCF result. An estimate, and labelled as one everywhere it travels."""

    entity_id: str
    period_label: str
    enterprise_value: str
    equity_value: str | None = None
    value_per_share: str | None = None
    currency: str
    #: The stated inputs. A valuation without them is not arguable, and a number
    #: nobody can argue with is a number nobody should act on.
    assumptions: dict[str, str] = Field(default_factory=dict)
    #: True when the terminal value dominates. The figure then rests mostly on
    #: the perpetuity assumption rather than on the forecast.
    terminal_dominated: bool = False


class ReportPayload(FrozenModel):
    """A research report's identity and verification verdict."""

    report_id: str
    entity_id: str
    period_label: str
    cited_source_ids: list[str] = Field(default_factory=list)
    numerically_grounded: bool
    citations_grounded: bool
    #: Figures the model introduced that Atlas did not compute. Empty on a
    #: published report by construction.
    unsupported_figures: list[str] = Field(default_factory=list)
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
    "ANALYSIS_COMPLETED",
    "FILING_DUPLICATE",
    "FILING_INGESTED",
    "PUBLISHED_EVENT_TYPES",
    "REPORT_PUBLISHED",
    "REPORT_WITHHELD",
    "SOURCE_APP",
    "STATEMENTS_EXTRACTED",
    "STATEMENTS_REJECTED",
    "VALUATION_PRODUCED",
    "AnalysisCompletedPayload",
    "FilingDuplicatePayload",
    "FilingIngestedPayload",
    "ReportPayload",
    "StatementsExtractedPayload",
    "StatementsRejectedPayload",
    "ValuationProducedPayload",
    "build_event",
]
