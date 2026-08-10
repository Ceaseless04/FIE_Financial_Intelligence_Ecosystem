"""Source documents and the chunks derived from them.

Chunks carry the character offsets they were cut from. That is what makes a
citation verifiable rather than decorative: given a chunk id, a reader can
recover the exact span of the original filing that supports a claim.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from fie_common.utils import content_hash, new_id, utc_now
from fie_schemas.base import FIEModel, FrozenModel
from fie_schemas.provenance import SourceLocator, SourceReference, SourceType


class DocumentType(StrEnum):
    SEC_FILING = "sec_filing"
    EARNINGS_CALL = "earnings_call"
    NEWS_ARTICLE = "news_article"
    PRESS_RELEASE = "press_release"
    MACRO_REPORT = "macro_report"


_DOCUMENT_TO_SOURCE: dict[DocumentType, SourceType] = {
    DocumentType.SEC_FILING: SourceType.SEC_FILING,
    DocumentType.EARNINGS_CALL: SourceType.EARNINGS_CALL,
    DocumentType.NEWS_ARTICLE: SourceType.NEWS_ARTICLE,
    DocumentType.PRESS_RELEASE: SourceType.NEWS_ARTICLE,
    DocumentType.MACRO_REPORT: SourceType.MACRO_INDICATOR,
}


class Document(FIEModel):
    """A source document awaiting or undergoing ingestion."""

    id: str = Field(default_factory=lambda: new_id("doc"))
    type: DocumentType
    title: str = Field(min_length=1, max_length=1024)
    content: str = Field(min_length=1)
    uri: str | None = None
    published_at: datetime | None = None
    ingested_at: datetime = Field(default_factory=utc_now)
    #: Free-form metadata from the source feed (ticker, CIK, form type...).
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def _content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document content must not be blank")
        return value

    @property
    def content_hash(self) -> str:
        """Stable hash of the content, used to detect re-ingestion.

        Feeds re-deliver the same filing routinely; hashing the content means a
        redelivery is recognised even when the feed assigns a new id.
        """
        return content_hash(self.content)

    @property
    def source_type(self) -> SourceType:
        return _DOCUMENT_TO_SOURCE[self.type]

    def to_source_reference(self, locator: SourceLocator | None = None) -> SourceReference:
        """Build a citable reference to this document, optionally to a span."""
        return SourceReference(
            source_id=self.id,
            source_type=self.source_type,
            title=self.title,
            uri=self.uri,
            published_at=self.published_at,
            locator=locator,
        )


def chunk_id_for(document_id: str, index: int) -> str:
    """Stable id for a chunk position within a document.

    Derived rather than random because re-ingestion must produce the same ids.
    A random id would mean the chunk row keeps its old identifier (the store
    upserts on document and index) while everything built in the same pass —
    mention rows, citations handed to other products — refers to a new one that
    does not exist.
    """
    digest = hashlib.sha256(f"{document_id}:{index}".encode()).hexdigest()
    return f"chk_{digest[:32]}"


class Chunk(FrozenModel):
    """A contiguous span of a document, sized for embedding and retrieval."""

    id: str = Field(default_factory=lambda: new_id("chk"))
    document_id: str = Field(min_length=1)
    #: Position within the document, 0-based.
    index: int = Field(ge=0)
    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    #: Heading path the chunk sits under, when the document has structure.
    section: str | None = None

    @model_validator(mode="after")
    def _validate_span(self) -> Chunk:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        if len(self.text) != self.end_char - self.start_char:
            raise ValueError(
                "chunk text length does not match its character span; "
                "offsets would not resolve back to the source document"
            )
        return self

    @property
    def locator(self) -> SourceLocator:
        return SourceLocator(
            start_char=self.start_char, end_char=self.end_char, section=self.section
        )

    def to_source_reference(self, document: Document) -> SourceReference:
        """Citation pointing at this exact span, with the supporting excerpt."""
        if document.id != self.document_id:
            raise ValueError("chunk does not belong to the supplied document")
        return SourceReference(
            source_id=document.id,
            source_type=document.source_type,
            title=document.title,
            uri=document.uri,
            published_at=document.published_at,
            locator=self.locator,
            excerpt=self.text[:1000],
        )

    def resolve_in(self, document: Document) -> str:
        """Re-read this chunk's span from the document.

        The round trip is what proves an offset is real; retrieval tests use it
        to confirm a citation still points at the text it claims to.
        """
        if document.id != self.document_id:
            raise ValueError("chunk does not belong to the supplied document")
        return document.content[self.start_char : self.end_char]


__all__ = ["Chunk", "Document", "DocumentType", "chunk_id_for"]
