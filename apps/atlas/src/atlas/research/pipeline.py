"""Producing a research report end to end.

The sequence below is the architecture in execution order, and it only runs one
way round:

1. Load the stored statements — facts, read from a filing.
2. Compute every ratio deterministically.
3. Optionally value the company, recording the assumptions the estimate rests
   on rather than presenting it as a measurement.
4. Fetch company context from MarketMind, and carry on without it if that
   fails.
5. Only now call the model, handing it the figures and asking what they mean.
6. Verify every number in the prose against what was computed in steps 2-3.

A model is never consulted before step 5, and never about a quantity. By the
time it is called, every figure in the report already exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from atlas.analysis.dcf import DCFAssumptions, DCFModel, DCFValuation, project_cash_flows
from atlas.analysis.growth import free_cash_flow
from atlas.analysis.ratios import analyze
from atlas.analysis.results import AnalysisResult
from atlas.clients.marketmind import MarketMindClient
from atlas.domain.money import Money, Rate
from atlas.domain.statements import FinancialStatements
from atlas.events import (
    ANALYSIS_COMPLETED,
    REPORT_PUBLISHED,
    REPORT_WITHHELD,
    VALUATION_PRODUCED,
    AnalysisCompletedPayload,
    ReportPayload,
    ValuationProducedPayload,
    build_event,
)
from atlas.research.reports import ReportRequest, ReportService, ResearchReport
from atlas.storage.repository import ReportRepository, StatementRepository
from fie_common.errors import NotFoundError, ValidationError
from fie_events.bus import EventBus
from fie_observability.logging import get_logger
from fie_observability.tracing import traced
from fie_schemas.base import FrozenModel

logger = get_logger(__name__)


@dataclass
class ValuationInputs:
    """What a caller must decide before a DCF can run.

    Supplied rather than inferred. A discount rate is a judgement about risk, and
    Atlas picking one silently would turn an arguable estimate into an apparent
    measurement — the precise failure the provenance rules exist to prevent.
    """

    discount_rate: Rate
    terminal_growth: Rate
    projection_years: int
    #: Growth applied to the base free cash flow across the forecast. Also a
    #: judgement, also stated.
    forecast_growth: Rate


class ReportOutcome(FrozenModel):
    """A generated report and where it was stored."""

    report_id: str
    report: ResearchReport
    analysis: AnalysisResult
    valuation: DCFValuation | None = None


@dataclass
class ResearchPipeline:
    """Assembles, verifies, stores, and announces a research report."""

    statements: StatementRepository
    reports: ReportRepository
    service: ReportService
    marketmind: MarketMindClient | None = None
    bus: EventBus | None = None

    async def produce(
        self,
        entity_id: str,
        *,
        company_name: str | None = None,
        fiscal_year: int | None = None,
        valuation_inputs: ValuationInputs | None = None,
        include_graph_context: bool = True,
    ) -> ReportOutcome:
        """Write a verified research note for a company's latest filed period.

        Raises:
            NotFoundError: if no statements are on file for the entity.
            ValidationError: if the statements support no computable metric.
        """
        statements = await self.load_statements(entity_id, fiscal_year)
        analysis = analyze(statements)

        if not analysis.available:
            raise ValidationError(
                "the filed statements support no computable metric, so there is "
                "nothing for a report to be written from",
                details={"entity_id": entity_id, "period": statements.period.label},
            )

        await self._publish(
            ANALYSIS_COMPLETED,
            AnalysisCompletedPayload(
                entity_id=entity_id,
                period_label=analysis.period_label,
                metric_names=[metric.name for metric in analysis.available],
                unavailable_names=[metric.name for metric in analysis.unavailable],
                source_ids=statements.source_ids,
            ),
        )

        valuation = (
            self.value(entity_id, statements, valuation_inputs)
            if valuation_inputs is not None
            else None
        )
        if valuation is not None:
            await self._publish(
                VALUATION_PRODUCED,
                ValuationProducedPayload(
                    entity_id=entity_id,
                    period_label=analysis.period_label,
                    enterprise_value=str(valuation.enterprise_value.amount),
                    equity_value=(
                        str(valuation.equity_value.amount) if valuation.equity_value else None
                    ),
                    value_per_share=(
                        str(valuation.value_per_share.amount) if valuation.value_per_share else None
                    ),
                    currency=valuation.currency,
                    assumptions=dict(valuation.assumptions),
                    terminal_dominated=valuation.is_terminal_dominated,
                ),
            )

        name = company_name or entity_id
        graph_context: list[str] = []
        if include_graph_context and self.marketmind is not None:
            graph_context = await self.marketmind.context_for(entity_id)

        with traced("atlas.research.pipeline", attributes={"entity_id": entity_id}):
            report = await self.service.generate(
                ReportRequest(
                    entity_id=entity_id,
                    company_name=name,
                    statements=statements,
                    analysis=analysis,
                    valuation=valuation,
                    graph_context=graph_context,
                    sources=statements.all_sources(),
                )
            )

        report_id = await self.reports.save(report)

        # Two event types rather than one with a flag. A consumer must opt into
        # the ungrounded case deliberately instead of missing a boolean.
        await self._publish(
            REPORT_PUBLISHED if report.is_publishable else REPORT_WITHHELD,
            ReportPayload(
                report_id=report_id,
                entity_id=entity_id,
                period_label=report.period_label,
                cited_source_ids=list(report.cited_source_ids),
                numerically_grounded=report.numerically_grounded,
                citations_grounded=report.citations_grounded,
                unsupported_figures=list(report.unsupported_figures),
                regeneration_count=report.regeneration_count,
            ),
        )

        return ReportOutcome(
            report_id=report_id, report=report, analysis=analysis, valuation=valuation
        )

    async def load_statements(
        self, entity_id: str, fiscal_year: int | None = None
    ) -> FinancialStatements:
        """The stored statements for a period, or the latest on file.

        Part of the interface rather than a private helper: the valuation
        endpoint needs exactly this lookup, and duplicating it there would let
        the two disagree about which period "latest" means.

        Raises:
            NotFoundError: if nothing is on file for the entity.
        """
        statements = (
            await self.statements.latest(entity_id)
            if fiscal_year is None
            else await self._for_year(entity_id, fiscal_year)
        )
        if statements is None:
            raise NotFoundError(
                "no financial statements are on file for this entity",
                details={"entity_id": entity_id, "fiscal_year": fiscal_year},
            )
        return statements

    async def _for_year(self, entity_id: str, fiscal_year: int) -> FinancialStatements | None:
        for candidate in await self.statements.history(entity_id, limit=40):
            if candidate.fiscal_year == fiscal_year:
                return candidate
        return None

    @staticmethod
    def value(
        entity_id: str, statements: FinancialStatements, inputs: ValuationInputs
    ) -> DCFValuation | None:
        """Run a DCF from the filed cash flow, or return nothing.

        Returning ``None`` rather than raising when free cash flow is
        unavailable: a company whose filing does not break out capital
        expenditure can still have a research report written about it, and
        inventing a base cash flow to force a valuation would be exactly the
        fabrication this product refuses.
        """
        base = free_cash_flow(statements)
        if not base.is_available:
            logger.info(
                "valuation_skipped",
                entity_id=entity_id,
                reason=base.unavailable_reason,
            )
            return None

        base_money = Money(amount=base.value, currency=statements.currency)
        if base_money.amount <= 0:
            logger.info("valuation_skipped", entity_id=entity_id, reason="negative free cash flow")
            return None

        assumptions = DCFAssumptions(
            discount_rate=inputs.discount_rate,
            terminal_growth_rate=inputs.terminal_growth,
            projection_years=inputs.projection_years,
        )
        flows = project_cash_flows(base_money, inputs.forecast_growth, inputs.projection_years)

        balance = statements.balance_sheet
        income = statements.income_statement
        return DCFModel(assumptions=assumptions, sources=statements.all_sources()).value(
            entity_id,
            flows,
            net_debt=balance.net_debt if balance is not None else None,
            diluted_shares=income.diluted_shares if income is not None else None,
        )

    async def _publish(self, event_type: str, payload: FrozenModel) -> None:
        if self.bus is None:
            return
        try:
            await self.bus.publish(build_event(event_type, payload))
        except Exception as error:  # noqa: BLE001 — an event must not fail the work
            logger.warning("event_publish_failed", event_type=event_type, error=str(error))


__all__ = ["ReportOutcome", "ResearchPipeline", "ValuationInputs"]
