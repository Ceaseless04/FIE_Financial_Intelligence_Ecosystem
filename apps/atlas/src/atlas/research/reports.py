"""Research report generation.

The division of labour, stated once and enforced below:

- Atlas computes every figure, deterministically, before the model is called.
- The model receives those figures and writes the analysis — what changed, what
  it suggests, what the risks are, what it cannot tell from the filing.
- Before the report is returned, every number in the prose is checked against
  the figures Atlas computed.

The model is never asked what a number is. It is asked what the numbers mean,
which is the part it is genuinely good at and the part deterministic code cannot
do.

A report that fails the numeric check is regenerated once with the offending
figures pointed out. If it fails again it is still returned — with
``is_publishable`` false and the unsupported figures listed — because silently
discarding the work would hide a systematic prompt or model problem behind an
empty response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from pydantic import Field

from atlas.analysis.dcf import DCFValuation
from atlas.domain.statements import FinancialStatements
from atlas.research.grounding import (
    NumericGroundingReport,
    allowed_values_from,
    check_numeric_grounding,
)
from fie_ai import AIRouter, CompletionRequest, Effort, Message, Role
from fie_ai.structured import check_grounding, parse_structured, structured_request
from fie_common.errors import AIProviderResponseError, ValidationError
from fie_finance.metrics import AnalysisResult, Metric
from fie_observability.logging import get_logger
from fie_observability.tracing import traced
from fie_schemas.base import FIEModel
from fie_schemas.provenance import Provenance, SourceReference

logger = get_logger(__name__)

REPORT_SYSTEM_PROMPT = """\
You are a financial analyst writing a research note.

Every figure you may use has already been computed and is given to you below. \
Your job is to explain what those figures mean, not to produce new ones.

Absolute rules:
- Never state a number that is not in the supplied figures. Do not compute, \
sum, average, annualise, or estimate any quantity yourself. If a figure you \
want is not supplied, write that it was not disclosed.
- You may round a supplied figure when writing it ($1,613,590,000 may be \
written as "$1.6 billion"). You may not adjust it.
- Cite the source_id for every factual claim drawn from a filing.
- Say plainly when a figure is an estimate rather than a reported fact, and \
name the assumption it rests on.
- Do not give investment advice. Do not recommend buying, selling, or holding, \
and do not state a price target. Describe what the figures show and what would \
change the picture.
- Where the data does not support a conclusion, say so. An honest gap is worth \
more than a confident guess."""

REGENERATION_NOTE = """\

Your previous draft contained figures that were not in the supplied data: {figures}.

Rewrite the report using only the supplied figures. Remove or rephrase every \
statement that depended on a number you introduced."""


class ReportSection(FIEModel):
    """One section of the narrative."""

    heading: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1)


class DraftReport(FIEModel):
    """The model's structured response."""

    summary: str = Field(min_length=1)
    sections: list[ReportSection] = Field(default_factory=list)
    #: Filing ids backing the factual claims.
    cited_source_ids: list[str] = Field(default_factory=list)
    #: What the filing does not answer. Valuable output in its own right.
    open_questions: list[str] = Field(default_factory=list)


class ResearchReport(FIEModel):
    """A verified research note."""

    entity_id: str
    period_label: str
    summary: str
    sections: list[ReportSection] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    #: The deterministic figures the narrative was written from.
    metrics: list[Metric] = Field(default_factory=list)
    cited_source_ids: list[str] = Field(default_factory=list)
    provenance: Provenance
    #: True when every figure in the prose traces to a computed value.
    numerically_grounded: bool = True
    #: Figures the model introduced that Atlas did not compute.
    unsupported_figures: list[str] = Field(default_factory=list)
    #: True when every citation resolved to a supplied source.
    citations_grounded: bool = True
    regeneration_count: int = Field(default=0, ge=0)

    @property
    def is_publishable(self) -> bool:
        """Whether this report may be shown to a reader as analysis.

        Both checks must pass. A report quoting an invented figure is worse than
        no report, because it is indistinguishable from a correct one.
        """
        return self.numerically_grounded and self.citations_grounded


@dataclass
class ReportRequest:
    """Everything a report is written from.

    Assembled by the caller so the figures are computed before the model is
    involved — the ordering is the architecture.
    """

    entity_id: str
    company_name: str
    statements: FinancialStatements
    analysis: AnalysisResult
    valuation: DCFValuation | None = None
    #: Prior-period metrics, so the narrative can discuss change.
    comparison: AnalysisResult | None = None
    #: Context from MarketMind: competitors, suppliers, executives.
    graph_context: list[str] = field(default_factory=list)
    sources: list[SourceReference] = field(default_factory=list)


@dataclass
class ReportService:
    """Turns computed analysis into a written, verified research note."""

    router: AIRouter
    max_tokens: int = 4096
    effort: Effort | None = Effort.HIGH
    #: How many times to regenerate when the numeric check fails.
    max_regenerations: int = 1

    async def generate(self, request: ReportRequest) -> ResearchReport:
        """Write and verify a research note.

        Raises:
            ValidationError: if no metrics were computed. A report written from
                nothing would be entirely model-authored, which is the outcome
                this whole design exists to prevent.
            AIProviderResponseError: if the model refuses or returns output that
                does not satisfy the schema.
        """
        if not request.analysis.available:
            raise ValidationError(
                "cannot write a research report with no computed figures",
                details={"entity_id": request.entity_id},
            )

        allowed = self._allowed_values(request)
        source_ids = [source.source_id for source in request.sources] or (
            request.statements.source_ids
        )

        with traced(
            "atlas.research.generate",
            attributes={
                "entity_id": request.entity_id,
                "metrics": len(request.analysis.available),
            },
        ):
            draft, grounding, regenerations = await self._draft_until_grounded(request, allowed)

        citation_report = check_grounding(
            draft.cited_source_ids, source_ids, require_citation=False
        )
        verified_citations = [
            source_id
            for source_id in draft.cited_source_ids
            if source_id not in citation_report.unknown_ids
        ]

        if grounding.unsupported:
            logger.warning(
                "report_numerically_ungrounded",
                entity_id=request.entity_id,
                unsupported=[figure.text for figure in grounding.unsupported],
                regenerations=regenerations,
            )

        return ResearchReport(
            entity_id=request.entity_id,
            period_label=request.analysis.period_label,
            summary=draft.summary,
            sections=draft.sections,
            open_questions=draft.open_questions,
            metrics=request.analysis.available,
            cited_source_ids=verified_citations,
            provenance=Provenance.generated(self.router.primary.default_model, *request.sources),
            numerically_grounded=grounding.is_grounded,
            unsupported_figures=[figure.text for figure in grounding.unsupported],
            citations_grounded=not citation_report.unknown_ids,
            regeneration_count=regenerations,
        )

    async def _draft_until_grounded(
        self, request: ReportRequest, allowed: set[Decimal]
    ) -> tuple[DraftReport, NumericGroundingReport, int]:
        """Draft, check the arithmetic, and retry once with the errors named."""
        prompt = self._build_prompt(request)
        draft = await self._draft(prompt)
        grounding = check_numeric_grounding(self._narrative_of(draft), allowed)

        attempts = 0
        while grounding.unsupported and attempts < self.max_regenerations:
            attempts += 1
            # Naming the offending figures makes the retry corrective rather
            # than a reroll of the same dice.
            retry_prompt = prompt + REGENERATION_NOTE.format(
                figures=", ".join(figure.text for figure in grounding.unsupported)
            )
            draft = await self._draft(retry_prompt)
            grounding = check_numeric_grounding(self._narrative_of(draft), allowed)

        return draft, grounding, attempts

    async def _draft(self, prompt: str) -> DraftReport:
        request = structured_request(
            CompletionRequest(
                messages=[Message(role=Role.USER, content=prompt)],
                system=REPORT_SYSTEM_PROMPT,
                max_tokens=self.max_tokens,
                effort=self.effort,
            ),
            DraftReport,
        )
        response = await self.router.complete(request)
        try:
            return parse_structured(response, DraftReport)
        except AIProviderResponseError:
            logger.warning("report_draft_failed")
            raise

    @staticmethod
    def _narrative_of(draft: DraftReport) -> str:
        """Every piece of prose a reader will see, checked as one body of text."""
        parts = [draft.summary]
        parts.extend(f"{section.heading}\n{section.body}" for section in draft.sections)
        parts.extend(draft.open_questions)
        return "\n\n".join(parts)

    @staticmethod
    def _allowed_values(request: ReportRequest) -> set[Decimal]:
        """Every figure the narrative is permitted to quote.

        Both the computed metrics and the reported figures they came from: a
        report should be free to state the revenue it computed a margin from.
        """
        computed = request.analysis.numeric_values()
        if request.comparison is not None:
            computed |= request.comparison.numeric_values()

        reported: set[Decimal] = set()
        statements = request.statements
        for statement in (
            statements.income_statement,
            statements.balance_sheet,
            statements.cash_flow_statement,
        ):
            if statement is None:
                continue
            for value in statement.model_dump().values():
                if isinstance(value, dict) and "amount" in value:
                    reported.add(Decimal(str(value["amount"])))

        valuation_values: set[Decimal] = set()
        if request.valuation is not None:
            valuation = request.valuation
            valuation_values = {
                valuation.enterprise_value.amount,
                valuation.present_value_of_forecast.amount,
                valuation.terminal_value.amount,
                valuation.present_value_of_terminal.amount,
            }
            if valuation.equity_value is not None:
                valuation_values.add(valuation.equity_value.amount)
            if valuation.value_per_share is not None:
                valuation_values.add(valuation.value_per_share.amount)

        return allowed_values_from(computed, reported, valuation_values)

    def _build_prompt(self, request: ReportRequest) -> str:
        """Render the computed figures for the model to write from."""
        parts = [
            f"Company: {request.company_name} (entity id: {request.entity_id})",
            f"Period: {request.analysis.period_label}",
            "",
            "## Computed figures",
            "These are the only numbers you may use.",
            "",
        ]

        for metric in request.analysis.available:
            inputs = (
                f"  (from {', '.join(f'{k}={v}' for k, v in metric.inputs.items())})"
                if metric.inputs
                else ""
            )
            parts.append(f"- {metric.name}: {metric.rendered()}{inputs}")

        unavailable = request.analysis.unavailable
        if unavailable:
            parts.extend(["", "## Not available", ""])
            parts.extend(f"- {metric.name}: {metric.unavailable_reason}" for metric in unavailable)

        if request.comparison is not None and request.comparison.available:
            parts.extend(["", f"## Prior period ({request.comparison.period_label})", ""])
            parts.extend(
                f"- {metric.name}: {metric.rendered()}" for metric in request.comparison.available
            )

        if request.valuation is not None:
            valuation = request.valuation
            parts.extend(
                [
                    "",
                    "## Valuation (an estimate, not a reported fact)",
                    "",
                    f"- enterprise_value: {valuation.enterprise_value}",
                ]
            )
            if valuation.equity_value is not None:
                parts.append(f"- equity_value: {valuation.equity_value}")
            if valuation.value_per_share is not None:
                parts.append(f"- value_per_share: {valuation.value_per_share}")
            parts.append(
                "- assumptions: " + ", ".join(f"{k}={v}" for k, v in valuation.assumptions.items())
            )
            if valuation.is_terminal_dominated:
                parts.append(
                    "- note: most of this value comes from the terminal value, so it "
                    "rests heavily on the perpetuity assumption. Say so."
                )

        if request.graph_context:
            parts.extend(["", "## Company context from the knowledge graph", ""])
            parts.extend(f"- {fact}" for fact in request.graph_context)

        if request.sources:
            parts.extend(["", "## Citable sources", ""])
            parts.extend(
                f'- source_id="{source.source_id}": {source.title or "filing"}'
                for source in request.sources
            )

        parts.extend(
            [
                "",
                "Write the research note. Explain what these figures show, what "
                "changed, and what a reader should be cautious about. Use only the "
                "figures above.",
            ]
        )
        return "\n".join(parts)


__all__ = [
    "REPORT_SYSTEM_PROMPT",
    "DraftReport",
    "ReportRequest",
    "ReportSection",
    "ReportService",
    "ResearchReport",
]
