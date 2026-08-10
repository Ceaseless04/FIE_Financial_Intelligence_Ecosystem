"""Deterministic metadata extraction from source documents.

Everything here is regex and rules — no model call. Tickers, CIKs, and form
types have exact syntax, and a deterministic extractor for them is faster,
free, reproducible, and auditable. Reserving the model for genuinely ambiguous
extraction is the same principle as keeping calculations out of the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from marketmind.domain.documents import Document
from marketmind.domain.entities import Identifier, IdentifierType

#: "(NASDAQ: ACME)", "NYSE:ACME", "(Nasdaq: ACME)"
_EXCHANGE_TICKER = re.compile(
    r"\((?:NASDAQ|NYSE|NYSE\s+American|AMEX|OTC|LSE|TSX)\s*[:\-]\s*([A-Z][A-Z0-9.\-]{0,9})\)",
    re.IGNORECASE,
)
_BARE_EXCHANGE_TICKER = re.compile(
    r"\b(?:NASDAQ|NYSE|AMEX|OTC|LSE|TSX)\s*[:\-]\s*([A-Z][A-Z0-9.\-]{0,9})\b"
)
#: "Commission File Number", "CIK 0000320193", "CIK: 320193"
_CIK = re.compile(r"\bCIK[\s:#]*0*(\d{1,10})\b", re.IGNORECASE)
#: SEC form designations.
_FORM_TYPE = re.compile(
    r"\b(?:FORM\s+)?(10-K|10-Q|8-K|20-F|40-F|S-1|S-3|DEF\s?14A|424B\d?)\b", re.IGNORECASE
)
#: Fiscal period statements.
_FISCAL_YEAR = re.compile(
    r"\b(?:fiscal\s+year|FY)\s*(?:ended|ending)?\s*[:\-]?\s*(\d{4})\b", re.IGNORECASE
)
_PERIOD_ENDED = re.compile(
    r"\bfor\s+the\s+(?:quarterly|annual|fiscal)\s+period\s+ended\s+"
    r"([A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
    re.IGNORECASE,
)
_LEI = re.compile(r"\bLEI[\s:#]*([A-Z0-9]{20})\b", re.IGNORECASE)

_MONTHS = {
    month: number
    for number, month in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        start=1,
    )
}

#: Words that look like tickers in all-caps text but are not.
_TICKER_STOPWORDS = frozenset(
    {"THE", "AND", "FOR", "INC", "LLC", "LTD", "CORP", "USA", "CEO", "CFO", "SEC", "GAAP"}
)


@dataclass
class DocumentMetadata:
    """Structured facts recovered from a document without a model call."""

    tickers: list[str] = field(default_factory=list)
    ciks: list[str] = field(default_factory=list)
    leis: list[str] = field(default_factory=list)
    form_type: str | None = None
    fiscal_year: int | None = None
    period_ended: date | None = None

    @property
    def identifiers(self) -> list[Identifier]:
        """Metadata expressed as entity identifiers.

        Invalid values are dropped rather than raised on: a malformed ticker in
        a noisy news article should not fail the whole ingestion.
        """
        results: list[Identifier] = []
        for identifier_type, values in (
            (IdentifierType.TICKER, self.tickers),
            (IdentifierType.CIK, self.ciks),
            (IdentifierType.LEI, self.leis),
        ):
            for value in values:
                try:
                    results.append(Identifier(type=identifier_type, value=value))
                except ValueError:
                    continue
        return results

    def to_dict(self) -> dict[str, object]:
        return {
            "tickers": self.tickers,
            "ciks": self.ciks,
            "leis": self.leis,
            "form_type": self.form_type,
            "fiscal_year": self.fiscal_year,
            "period_ended": self.period_ended.isoformat() if self.period_ended else None,
        }


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _parse_long_date(raw: str) -> date | None:
    match = re.match(r"([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})", raw.strip())
    if not match:
        return None
    month = _MONTHS.get(match.group(1).lower())
    if month is None:
        return None
    try:
        return date(int(match.group(3)), month, int(match.group(2)))
    except ValueError:
        return None


def extract_metadata(document: Document) -> DocumentMetadata:
    """Recover identifiers and filing metadata from a document."""
    content = document.content
    metadata = DocumentMetadata()

    tickers = [match.upper() for match in _EXCHANGE_TICKER.findall(content)]
    tickers += [match.upper() for match in _BARE_EXCHANGE_TICKER.findall(content)]
    metadata.tickers = _unique([ticker for ticker in tickers if ticker not in _TICKER_STOPWORDS])

    metadata.ciks = _unique([cik.lstrip("0") or "0" for cik in _CIK.findall(content)])
    metadata.leis = _unique([lei.upper() for lei in _LEI.findall(content)])

    form_match = _FORM_TYPE.search(content)
    if form_match:
        metadata.form_type = re.sub(r"\s+", "", form_match.group(1)).upper()

    fiscal_match = _FISCAL_YEAR.search(content)
    if fiscal_match:
        year = int(fiscal_match.group(1))
        # Reject years that cannot be a filing period; a four-digit number in
        # running text is more often a dollar figure than a fiscal year.
        if 1900 <= year <= 2100:
            metadata.fiscal_year = year

    period_match = _PERIOD_ENDED.search(content)
    if period_match:
        metadata.period_ended = _parse_long_date(period_match.group(1))
        if metadata.period_ended and metadata.fiscal_year is None:
            metadata.fiscal_year = metadata.period_ended.year

    # Feed-supplied metadata is authoritative over anything scraped from prose.
    for key, target in (("ticker", "tickers"), ("cik", "ciks")):
        supplied = document.metadata.get(key)
        if isinstance(supplied, str) and supplied.strip():
            current: list[str] = getattr(metadata, target)
            value = supplied.strip().upper() if key == "ticker" else supplied.strip().lstrip("0")
            setattr(metadata, target, _unique([value, *current]))

    return metadata


__all__ = ["DocumentMetadata", "extract_metadata"]
