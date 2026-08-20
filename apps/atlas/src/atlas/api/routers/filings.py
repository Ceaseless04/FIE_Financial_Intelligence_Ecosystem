"""Filing ingestion and retrieval."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status

from atlas.api.dependencies import (
    PERMISSION_READ_FILING,
    PERMISSION_WRITE_FILING,
    ContainerDep,
    SessionDep,
    require_permission,
)
from atlas.api.schemas import (
    FilingView,
    IngestFilingRequest,
    IngestFilingResponse,
    StatementSetView,
)
from fie_common.errors import NotFoundError
from fie_finance.periods import PeriodKind

router = APIRouter(prefix="/api/v1", tags=["filings"])


@router.post(
    "/filings",
    response_model=IngestFilingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a filing and extract its statements",
    dependencies=[Depends(require_permission(PERMISSION_WRITE_FILING))],
)
async def ingest_filing(
    request: IngestFilingRequest, container: ContainerDep, session: SessionDep
) -> IngestFilingResponse:
    """Store a filing, transcribe its statements, and validate the arithmetic.

    Returns 201 whether or not statements were extracted — the filing itself is
    stored either way, and losing a citable document because a table could not
    be read would be the worse outcome. Look at ``statement_set_id`` and
    ``rejected`` to tell the cases apart.

    A redelivered filing is recognised by content hash and returns
    ``newly_ingested: false`` without re-running extraction.
    """
    result = await container.ingestion(session).ingest(request.to_filing())
    return IngestFilingResponse.from_result(result)


@router.get(
    "/filings/{filing_id}",
    response_model=FilingView,
    summary="Fetch a stored filing",
    dependencies=[Depends(require_permission(PERMISSION_READ_FILING))],
)
async def get_filing(filing_id: str, container: ContainerDep, session: SessionDep) -> FilingView:
    """Retrieve a filing's metadata.

    Raises:
        NotFoundError: if no such filing is stored.
    """
    filing = await container.filings(session).get(filing_id)
    if filing is None:
        raise NotFoundError("no such filing", details={"filing_id": filing_id})
    return FilingView.from_filing(filing)


@router.get(
    "/entities/{entity_id}/filings",
    response_model=list[FilingView],
    summary="List filings for a company",
    dependencies=[Depends(require_permission(PERMISSION_READ_FILING))],
)
async def list_filings(
    entity_id: str,
    container: ContainerDep,
    session: SessionDep,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[FilingView]:
    """Filings on record for one entity, most recent period first."""
    filings = await container.filings(session).list_for_entity(entity_id, limit=limit)
    return [FilingView.from_filing(filing) for filing in filings]


@router.get(
    "/entities/{entity_id}/statements",
    response_model=StatementSetView,
    summary="Fetch extracted statements",
    dependencies=[Depends(require_permission(PERMISSION_READ_FILING))],
)
async def get_statements(
    entity_id: str,
    container: ContainerDep,
    session: SessionDep,
    fiscal_year: int | None = Query(default=None, ge=1900, le=2200),
    kind: PeriodKind = Query(default=PeriodKind.ANNUAL),
    quarter: int | None = Query(default=None, ge=1, le=4),
) -> StatementSetView:
    """Validated figures for a period, or the latest on file.

    Every amount is returned as a string, and a line item the filing did not
    disclose is ``null`` rather than ``0``.

    Raises:
        NotFoundError: if no statements are stored for the period requested.
    """
    repository = container.statements(session)
    statements = (
        await repository.latest(entity_id)
        if fiscal_year is None
        else await repository.get_for_period(
            entity_id, fiscal_year=fiscal_year, kind=kind, quarter=quarter
        )
    )
    if statements is None:
        raise NotFoundError(
            "no financial statements are on file for this entity and period",
            details={"entity_id": entity_id, "fiscal_year": fiscal_year},
        )
    return StatementSetView.from_statements(statements)


__all__ = ["router"]
