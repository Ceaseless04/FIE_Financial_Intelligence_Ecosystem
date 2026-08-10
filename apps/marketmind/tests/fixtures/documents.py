"""Sample documents used across the MarketMind test suites.

Written to look like the real thing — headings, exchange tickers, CIK lines,
period statements — because the chunker splits on structure and the metadata
extractor keys on those exact conventions. Prose that reads plausibly but has no
filing structure would exercise neither.

Nothing here is real. The companies are invented so no test can accidentally
assert a claim about an actual issuer.
"""

from __future__ import annotations

from datetime import UTC, datetime

from marketmind.domain.documents import Document, DocumentType

ACME_10K = """\
# ANNUAL REPORT ON FORM 10-K

Acme Robotics Corporation
(NASDAQ: ACME)
CIK: 0000123456

For the fiscal period ended December 31, 2025

## ITEM 1. BUSINESS

Acme Robotics Corporation designs and manufactures industrial automation \
systems for automotive and logistics customers. The Company was incorporated in \
Delaware and is headquartered in Detroit, Michigan.

Sofia Marchetti has served as Chief Executive Officer since March 2021. \
Daniel Okonkwo serves as Chief Financial Officer.

The Company operates in the industrial automation industry. Its principal \
product line is the Atlas-7 robotic arm.

## ITEM 1A. RISK FACTORS

The Company depends on Northwind Components Ltd for precision bearings used in \
substantially all of its products. A disruption at Northwind Components Ltd \
would adversely affect the Company's ability to manufacture.

The Company competes with Zenith Automation Inc in the industrial automation \
market. Competition is based on price, reliability, and service.

## ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

Revenue for fiscal 2025 was $412.6 million, compared with $358.1 million in \
fiscal 2024. The increase was driven by higher unit volumes in the logistics \
segment.
"""

ACME_NEWS = """\
Acme Robotics names new operations chief

Acme Robotics Corporation (NASDAQ: ACME) announced today that it has appointed \
Priya Raghavan as Chief Operating Officer, effective immediately.

Acme Robotics Corp said the appointment supports its expansion in the logistics \
automation market, where it competes with Zenith Automation Inc.
"""

NORTHWIND_FILING = """\
# ANNUAL REPORT ON FORM 10-K

Northwind Components Ltd
(NYSE: NWC)
CIK: 0000987654

For the fiscal period ended December 31, 2025

## ITEM 1. BUSINESS

Northwind Components Ltd manufactures precision bearings and linear motion \
components. The company is headquartered in Cleveland, Ohio.

Northwind Components Ltd supplies precision bearings to Acme Robotics \
Corporation and to several other industrial customers.
"""

#: A document with no headings at all, to exercise the chunker's fallback path.
UNSTRUCTURED_NOTE = (
    "Quarterly commentary. " * 40
    + "\n\n"
    + "The logistics segment grew faster than the automotive segment. " * 30
)


def acme_10k() -> Document:
    return Document(
        type=DocumentType.SEC_FILING,
        title="Acme Robotics Corporation — Form 10-K (FY2025)",
        content=ACME_10K,
        uri="https://example.invalid/filings/acme-10k-2025",
        published_at=datetime(2026, 2, 18, tzinfo=UTC),
        metadata={"ticker": "ACME"},
    )


def acme_news() -> Document:
    return Document(
        type=DocumentType.NEWS_ARTICLE,
        title="Acme Robotics names new operations chief",
        content=ACME_NEWS,
        uri="https://example.invalid/news/acme-coo",
        published_at=datetime(2026, 3, 2, tzinfo=UTC),
    )


def northwind_10k() -> Document:
    return Document(
        type=DocumentType.SEC_FILING,
        title="Northwind Components Ltd — Form 10-K (FY2025)",
        content=NORTHWIND_FILING,
        uri="https://example.invalid/filings/northwind-10k-2025",
        published_at=datetime(2026, 2, 20, tzinfo=UTC),
    )


def unstructured_note() -> Document:
    return Document(
        type=DocumentType.MACRO_REPORT,
        title="Unstructured quarterly note",
        content=UNSTRUCTURED_NOTE,
    )


__all__ = [
    "ACME_10K",
    "ACME_NEWS",
    "NORTHWIND_FILING",
    "UNSTRUCTURED_NOTE",
    "acme_10k",
    "acme_news",
    "northwind_10k",
    "unstructured_note",
]
