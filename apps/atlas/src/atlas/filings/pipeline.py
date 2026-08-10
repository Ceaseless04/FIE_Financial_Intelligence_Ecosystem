"""Filing ingestion: store, transcribe, validate, persist, announce.

The ordering is the design. The filing is stored *before* extraction runs, so a
transcription that fails still leaves a citable document on record rather than
losing both. Extraction then either produces statements that satisfy the
accounting identities or it produces reasons — and both outcomes are published,
because a filing whose balance sheet will not balance is a fact other products
want to know.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field

from atlas.events import (
    FILING_DUPLICATE,
    FILING_INGESTED,
    STATEMENTS_EXTRACTED,
    STATEMENTS_REJECTED,
    FilingDuplicatePayload,
    FilingIngestedPayload,
    StatementsExtractedPayload,
    StatementsRejectedPayload,
    build_event,
)
from atlas.filings.extraction import StatementExtractor
from atlas.filings.models import Filing
from atlas.storage.repository import FilingRepository, StatementRepository
from fie_events.bus import EventBus
from fie_observability.logging import get_logger
from fie_observability.tracing import traced
from fie_schemas.base import FrozenModel

logger = get_logger(__name__)


class IngestionResult(FrozenModel):
    """What one filing produced."""

    filing_id: str
    entity_id: str
    period_label: str
    #: False when the filing was already on record; nothing was re-extracted.
    newly_ingested: bool = True
    statement_set_id: str | None = None
    has_income_statement: bool = False
    has_balance_sheet: bool = False
    has_cash_flow_statement: bool = False
    #: Everything the extractor refused, whether or not a usable set survived.
    rejected: list[str] = Field(default_factory=list)
    #: True when figures were transcribed but contradicted each other.
    failed_validation: bool = False

    @property
    def has_statements(self) -> bool:
        return self.statement_set_id is not None


@dataclass
class FilingPipeline:
    """Turns a source filing into stored, validated financial statements."""

    filings: FilingRepository
    statements: StatementRepository
    extractor: StatementExtractor
    bus: EventBus | None = None

    async def ingest(self, filing: Filing) -> IngestionResult:
        """Store a filing and extract its statements.

        A redelivered filing is recognised by content hash and returns early:
        re-extracting it would spend a model call reproducing a result already
        held, and could produce a *different* one, which is worse.
        """
        with traced(
            "atlas.filings.ingest",
            attributes={"entity_id": filing.entity_id, "filing_type": str(filing.type)},
        ):
            filing_id, created = await self.filings.upsert(filing)

            if not created:
                await self._publish(
                    FILING_DUPLICATE,
                    FilingDuplicatePayload(
                        filing_id=filing.id,
                        existing_filing_id=filing_id,
                        entity_id=filing.entity_id,
                        content_hash=filing.content_hash,
                    ),
                )
                logger.info("filing_duplicate", filing_id=filing.id, existing_filing_id=filing_id)
                return IngestionResult(
                    filing_id=filing_id,
                    entity_id=filing.entity_id,
                    period_label=filing.period.label,
                    newly_ingested=False,
                )

            await self._publish(
                FILING_INGESTED,
                FilingIngestedPayload(
                    filing_id=filing_id,
                    entity_id=filing.entity_id,
                    filing_type=str(filing.type),
                    period_label=filing.period.label,
                    title=filing.title,
                    content_hash=filing.content_hash,
                    uri=filing.uri,
                ),
            )

            outcome = await self.extractor.extract(filing)

            if outcome.statements is None:
                await self._publish(
                    STATEMENTS_REJECTED,
                    StatementsRejectedPayload(
                        filing_id=filing_id,
                        entity_id=filing.entity_id,
                        period_label=filing.period.label,
                        reasons=list(outcome.rejected),
                        failed_validation=outcome.failed_validation,
                    ),
                )
                logger.warning(
                    "statements_not_extracted",
                    filing_id=filing_id,
                    rejected=outcome.rejection_count,
                )
                return IngestionResult(
                    filing_id=filing_id,
                    entity_id=filing.entity_id,
                    period_label=filing.period.label,
                    rejected=list(outcome.rejected),
                    failed_validation=outcome.failed_validation,
                )

            statements = outcome.statements
            statement_set_id = await self.statements.save(
                statements, filing_id=filing_id, rejected=outcome.rejected
            )

            await self._publish(
                STATEMENTS_EXTRACTED,
                StatementsExtractedPayload(
                    statement_set_id=statement_set_id,
                    filing_id=filing_id,
                    entity_id=filing.entity_id,
                    period_label=filing.period.label,
                    currency=statements.currency,
                    has_income_statement=statements.income_statement is not None,
                    has_balance_sheet=statements.balance_sheet is not None,
                    has_cash_flow_statement=statements.cash_flow_statement is not None,
                    rejected=list(outcome.rejected),
                ),
            )

            return IngestionResult(
                filing_id=filing_id,
                entity_id=filing.entity_id,
                period_label=filing.period.label,
                statement_set_id=statement_set_id,
                has_income_statement=statements.income_statement is not None,
                has_balance_sheet=statements.balance_sheet is not None,
                has_cash_flow_statement=statements.cash_flow_statement is not None,
                rejected=list(outcome.rejected),
                failed_validation=outcome.failed_validation,
            )

    async def _publish(self, event_type: str, payload: FrozenModel) -> None:
        """Publish, treating a bus failure as non-fatal.

        The work is already committed by the time an event is emitted. Failing
        the request because the announcement did not land would roll back a
        successful extraction over a messaging problem.
        """
        if self.bus is None:
            return
        try:
            await self.bus.publish(build_event(event_type, payload))
        except Exception as error:  # noqa: BLE001 — an event must not fail the work
            logger.warning("event_publish_failed", event_type=event_type, error=str(error))


__all__ = ["FilingPipeline", "IngestionResult"]
