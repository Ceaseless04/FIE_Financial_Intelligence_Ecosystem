"""Management commentary: what the variances mean, written by a model.

The division of labour is the one the ecosystem runs on, with one addition that
is specific to planning.

Atlas verifies that every *figure* in a research note is one it computed. A
variance commentary needs that, and needs one thing more: that every *claim
about direction* matches what the numbers actually did. "Marketing delivered
real savings" against a hundred-thousand overspend contains no wrong number.
Numeric grounding passes it. A reader believes it.

So a commentary must clear both gates before it is publishable:

1. **Numeric grounding** — every figure traces to a computed value.
2. **Direction grounding** — no sentence asserts a direction the computed
   variance refutes.

An ungrounded draft is regenerated once, with the specific problems named, and
if it fails again it is returned marked unpublishable rather than discarded.
Throwing it away would hide a systematic prompt problem behind an empty
response; returning it unlabelled would be worse than either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from pydantic import Field

from cfo_ai.analysis.cash import CashPosition
from cfo_ai.analysis.forecast import Forecast
from cfo_ai.analysis.variance import VarianceReport
from cfo_ai.domain.accounts import Account
from cfo_ai.research.direction import DirectionGroundingReport, check_direction_grounding
from fie_ai import AIRouter, CompletionRequest, Effort, Message, Role
from fie_ai.structured import parse_structured, structured_request
from fie_common.errors import AIProviderResponseError, ValidationError
from fie_finance.grounding import (
    NumericGroundingReport,
    allowed_values_from,
    check_numeric_grounding,
)
from fie_finance.metrics import Metric
from fie_observability.logging import get_logger
from fie_observability.tracing import traced
from fie_schemas.base import FIEModel
from fie_schemas.provenance import Provenance, SourceReference

logger = get_logger(__name__)

COMMENTARY_SYSTEM_PROMPT = """\
You are a financial planning and analysis lead writing management commentary on \
a budget-versus-actual report.

Every figure you may use has already been computed and is given to you below, \
and so has the direction of every variance. Your job is to explain what \
happened and what it means, not to produce figures or to decide which way a \
line moved.

Absolute rules:
- Never state a number that is not in the supplied figures. Do not compute, \
sum, average, annualise, or estimate any quantity yourself.
- You may round a supplied figure when writing it. You may not adjust it.
- **Use the direction exactly as supplied.** Each line states whether it came in \
above or below plan and whether that is favourable or unfavourable. A cost \
below plan is favourable; revenue below plan is not. Do not infer this from the \
sign of a number — it is stated for you, and inferring it is the most common way \
this kind of commentary goes wrong.
- Where a line is marked "direction not assessed", do not call it good or bad. \
Report the movement and leave the judgement to the reader.
- Forecasts and runway figures are estimates resting on stated assumptions. Say \
so, and name the assumption.
- Do not recommend headcount actions, and do not attribute a variance to a named \
individual.
- Where the data does not support a conclusion, say so. An honest gap is worth \
more than a confident guess."""

REGENERATION_NOTE = """\

Your previous draft had the following problems:
{problems}

Rewrite the commentary. Use only the supplied figures, and use the supplied \
direction for every line rather than inferring it."""


class CommentarySection(FIEModel):
    """One section of the narrative."""

    heading: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1)


class DraftCommentary(FIEModel):
    """The model's structured response."""

    summary: str = Field(min_length=1)
    sections: list[CommentarySection] = Field(default_factory=list)
    #: What the writer could not determine from the data.
    open_questions: list[str] = Field(default_factory=list)


class Commentary(FIEModel):
    """A verified management commentary."""

    entity_id: str
    period_label: str
    summary: str
    sections: list[CommentarySection] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    #: The deterministic figures the narrative was written from.
    metrics: list[Metric] = Field(default_factory=list)
    provenance: Provenance
    numerically_grounded: bool = True
    unsupported_figures: list[str] = Field(default_factory=list)
    #: True when no sentence contradicts a computed variance direction.
    directionally_grounded: bool = True
    #: Sentences that reversed the meaning of a line, with an explanation each.
    direction_contradictions: list[str] = Field(default_factory=list)
    regeneration_count: int = Field(default=0, ge=0)

    @property
    def is_publishable(self) -> bool:
        """Both gates. A commentary that quotes a real number and draws the
        opposite conclusion from it is not a lesser failure than one that
        invents a number — it is a more persuasive one."""
        return self.numerically_grounded and self.directionally_grounded


@dataclass
class CommentaryRequest:
    """Everything a commentary is written from.

    Assembled by the caller so every figure exists before the model is involved.
    """

    entity_id: str
    company_name: str
    variances: VarianceReport
    accounts: dict[str, Account]
    cash: CashPosition | None = None
    forecasts: list[Forecast] = field(default_factory=list)
    sources: list[SourceReference] = field(default_factory=list)


@dataclass
class CommentaryService:
    """Turns a variance report into written, doubly-verified commentary."""

    router: AIRouter
    max_tokens: int = 3072
    effort: Effort | None = Effort.HIGH
    max_regenerations: int = 1

    async def generate(self, request: CommentaryRequest) -> Commentary:
        """Write and verify management commentary.

        Raises:
            ValidationError: if the report contains no variances. A commentary
                written from nothing would be entirely model-authored.
            AIProviderResponseError: if the model refuses or returns output that
                does not satisfy the schema.
        """
        if not request.variances.variances:
            raise ValidationError(
                "cannot write commentary with no computed variances",
                details={"entity_id": request.entity_id},
            )

        metrics = request.variances.to_metrics(request.sources)
        if request.cash is not None:
            metrics.extend(request.cash.to_metrics(request.sources))
        for forecast in request.forecasts:
            metrics.extend(forecast.to_metrics(request.sources))

        allowed = self._allowed_values(request, metrics)
        accounts = list(request.accounts.values())

        with traced(
            "cfo_ai.commentary.generate",
            attributes={
                "entity_id": request.entity_id,
                "variances": len(request.variances.variances),
            },
        ):
            draft, numeric, direction, regenerations = await self._draft_until_grounded(
                request, allowed, accounts
            )

        if numeric.unsupported or direction.contradictions:
            logger.warning(
                "commentary_ungrounded",
                entity_id=request.entity_id,
                unsupported_figures=[figure.text for figure in numeric.unsupported],
                direction_contradictions=len(direction.contradictions),
                regenerations=regenerations,
            )

        return Commentary(
            entity_id=request.entity_id,
            period_label=request.variances.period.label,
            summary=draft.summary,
            sections=draft.sections,
            open_questions=draft.open_questions,
            metrics=metrics,
            provenance=Provenance.generated(self.router.primary.default_model, *request.sources),
            numerically_grounded=numeric.is_grounded,
            unsupported_figures=[figure.text for figure in numeric.unsupported],
            directionally_grounded=direction.is_grounded,
            direction_contradictions=[
                contradiction.explanation for contradiction in direction.contradictions
            ],
            regeneration_count=regenerations,
        )

    async def _draft_until_grounded(
        self,
        request: CommentaryRequest,
        allowed: set[Decimal],
        accounts: list[Account],
    ) -> tuple[DraftCommentary, NumericGroundingReport, DirectionGroundingReport, int]:
        """Draft, check both gates, and retry once with the failures named."""
        prompt = self._build_prompt(request)
        draft = await self._draft(prompt)
        numeric, direction = self._verify(draft, allowed, request, accounts)

        attempts = 0
        while (
            numeric.unsupported or direction.contradictions
        ) and attempts < self.max_regenerations:
            attempts += 1
            problems: list[str] = []
            if numeric.unsupported:
                figures = ", ".join(figure.text for figure in numeric.unsupported)
                problems.append(f"- these figures were not in the supplied data: {figures}")
            for contradiction in direction.contradictions:
                problems.append(f"- {contradiction.explanation}")

            draft = await self._draft(
                prompt + REGENERATION_NOTE.format(problems="\n".join(problems))
            )
            numeric, direction = self._verify(draft, allowed, request, accounts)

        return draft, numeric, direction, attempts

    def _verify(
        self,
        draft: DraftCommentary,
        allowed: set[Decimal],
        request: CommentaryRequest,
        accounts: list[Account],
    ) -> tuple[NumericGroundingReport, DirectionGroundingReport]:
        narrative = self._narrative_of(draft)
        return (
            check_numeric_grounding(narrative, allowed),
            check_direction_grounding(narrative, request.variances.variances, accounts),
        )

    async def _draft(self, prompt: str) -> DraftCommentary:
        request = structured_request(
            CompletionRequest(
                messages=[Message(role=Role.USER, content=prompt)],
                system=COMMENTARY_SYSTEM_PROMPT,
                max_tokens=self.max_tokens,
                effort=self.effort,
            ),
            DraftCommentary,
        )
        response = await self.router.complete(request)
        try:
            return parse_structured(response, DraftCommentary)
        except AIProviderResponseError:
            logger.warning("commentary_draft_failed")
            raise

    @staticmethod
    def _narrative_of(draft: DraftCommentary) -> str:
        parts = [draft.summary]
        parts.extend(f"{section.heading}\n{section.body}" for section in draft.sections)
        parts.extend(draft.open_questions)
        return "\n\n".join(parts)

    @staticmethod
    def _allowed_values(request: CommentaryRequest, metrics: list[Metric]) -> set[Decimal]:
        """Every figure the narrative may quote.

        The computed metrics, plus the plan and actual amounts they came from —
        a commentary should be free to state the budget it is discussing.
        """
        computed = {metric.value for metric in metrics if metric.is_available}
        reported: set[Decimal] = set()
        for variance in request.variances.variances:
            reported.add(variance.budget.amount)
            reported.add(variance.actual.amount)
            if variance.percent is not None:
                reported.add(variance.percent)
        if request.cash is not None:
            reported.add(request.cash.burn.opening_cash.amount)
            reported.add(request.cash.burn.closing_cash.amount)
        return allowed_values_from(computed, reported)

    def _build_prompt(self, request: CommentaryRequest) -> str:
        """Render the computed figures *and their directions* for the model."""
        report = request.variances
        parts = [
            f"Company: {request.company_name} (entity id: {request.entity_id})",
            f"Period: {report.period.label}",
            f"Currency: {report.currency}",
            "",
            "## Variances",
            "These are the only numbers you may use, and the direction of each "
            "line is stated. Use it as given.",
            "",
        ]
        parts.extend(f"- {variance.describe()}" for variance in report.variances)

        parts.extend(
            [
                "",
                f"Net operating variance: {report.net_operating_variance()} "
                "(revenue variances add, cost variances subtract; capital "
                "expenditure and cash movements are excluded).",
            ]
        )

        if report.not_assessed:
            parts.extend(
                [
                    "",
                    "## Lines with no direction",
                    "Report the movement on these and do not call it good or bad.",
                    "",
                ]
            )
            parts.extend(
                f"- {variance.account_name} [{variance.account_code}]"
                for variance in report.not_assessed
            )

        if report.unmatched_actual_keys:
            parts.extend(["", "## Spend with no budget line", ""])
            parts.extend(f"- {key}" for key in report.unmatched_actual_keys)

        if report.unmatched_budget_keys:
            parts.extend(["", "## Budget lines with no actual booked", ""])
            parts.extend(f"- {key}" for key in report.unmatched_budget_keys)

        if request.cash is not None:
            parts.extend(["", "## Cash (estimates, not measurements)", ""])
            parts.append(f"- {request.cash.runway.describe()}")

        for forecast in request.forecasts:
            parts.extend(
                [
                    "",
                    f"## Forecast for {forecast.account_code} (an estimate, not a reported fact)",
                    "",
                    f"- total over {len(forecast.points)} periods: {forecast.total}",
                    "- assumptions: "
                    + ", ".join(f"{key}={value}" for key, value in forecast.assumptions.items()),
                ]
            )

        parts.extend(
            [
                "",
                "Write the management commentary. Explain what happened, what it "
                "means, and what a reader should watch. Use only the figures "
                "above and the directions as stated.",
            ]
        )
        return "\n".join(parts)


__all__ = [
    "COMMENTARY_SYSTEM_PROMPT",
    "Commentary",
    "CommentaryRequest",
    "CommentarySection",
    "CommentaryService",
    "DraftCommentary",
]
