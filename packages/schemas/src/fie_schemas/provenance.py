"""Provenance contracts — the mechanism that keeps facts separable from analysis.

Four of the six products carry an explicit constraint about grounding: Atlas
must "separate factual data from generated analysis" and "cite source data";
Venture must "separate sourced facts from estimates" and "make valuation
assumptions explicit"; Sentinel and CFO.ai must not let an LLM stand in for a
deterministic calculation.

Rather than restate that as prose in six codebases, it is encoded here as
validation. A ``Provenance`` for a fact cannot be constructed without a source,
a derived value cannot be constructed without naming the deterministic service
that produced it, and an estimate cannot be constructed without its
assumptions. Grounding therefore fails loudly at construction time instead of
silently in a report.

This module contains no financial semantics — it is an attribution mechanism,
not domain logic.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import Field, model_validator

from fie_common.utils import utc_now
from fie_schemas.base import FrozenModel

T = TypeVar("T")


class AssertionKind(StrEnum):
    """How a value came to exist. Determines what evidence must accompany it."""

    #: Read directly from a source document or dataset. Requires citations.
    FACT = "fact"
    #: Computed by a deterministic service from facts. Requires the service name.
    DERIVED = "derived"
    #: Projection resting on assumptions. Requires those assumptions.
    ESTIMATE = "estimate"
    #: Narrative authored by a language model. Requires the model id.
    GENERATED = "generated"


class SourceType(StrEnum):
    """Category of an evidence source."""

    SEC_FILING = "sec_filing"
    EARNINGS_CALL = "earnings_call"
    NEWS_ARTICLE = "news_article"
    MARKET_DATA = "market_data"
    MACRO_INDICATOR = "macro_indicator"
    INTERNAL_DOCUMENT = "internal_document"
    PITCH_DECK = "pitch_deck"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    USER_INPUT = "user_input"
    COMPUTATION = "computation"
    MODEL_OUTPUT = "model_output"


class SourceLocator(FrozenModel):
    """Where inside a source the evidence sits, for verifiable citations."""

    page: int | None = Field(default=None, ge=1)
    section: str | None = None
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_range(self) -> SourceLocator:
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char < self.start_char
        ):
            raise ValueError("end_char must be >= start_char")
        return self


class SourceReference(FrozenModel):
    """A citable pointer back to primary evidence."""

    source_id: str = Field(min_length=1)
    source_type: SourceType
    title: str | None = None
    uri: str | None = None
    published_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)
    locator: SourceLocator | None = None
    excerpt: str | None = Field(
        default=None,
        description="Verbatim text supporting the assertion; never paraphrased.",
    )


class Provenance(FrozenModel):
    """Evidence backing a single value.

    The model validator is the enforcement point — see the module docstring.
    """

    kind: AssertionKind
    sources: list[SourceReference] = Field(default_factory=list)
    computation: str | None = Field(
        default=None,
        description="Name of the deterministic service that produced a DERIVED value.",
    )
    assumptions: dict[str, Any] = Field(
        default_factory=dict,
        description="Named inputs an ESTIMATE rests on. Must be explicit and displayable.",
    )
    model: str | None = Field(
        default=None, description="Model identifier for a GENERATED assertion."
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: str | None = None

    @model_validator(mode="after")
    def _validate_evidence(self) -> Provenance:
        if self.kind is AssertionKind.FACT:
            if not self.sources:
                raise ValueError("a FACT requires at least one source reference")
            if self.model is not None:
                raise ValueError("a FACT cannot name a model; model-authored content is GENERATED")
        elif self.kind is AssertionKind.DERIVED:
            if not self.computation:
                raise ValueError(
                    "a DERIVED value must name the deterministic computation that produced it"
                )
            if self.model is not None:
                raise ValueError(
                    "a DERIVED value must come from a deterministic service, not a model"
                )
        elif self.kind is AssertionKind.ESTIMATE:
            if not self.assumptions:
                raise ValueError("an ESTIMATE must state the assumptions it rests on")
        elif self.kind is AssertionKind.GENERATED:
            if not self.model:
                raise ValueError("a GENERATED assertion must name the model that produced it")
        return self

    @property
    def is_model_authored(self) -> bool:
        """True when a language model produced the value.

        Presentation layers use this to label output rather than inferring it.
        """
        return self.kind is AssertionKind.GENERATED

    @property
    def is_verifiable(self) -> bool:
        """True when the value traces back to primary evidence or a computation."""
        return bool(self.sources) or bool(self.computation)

    @classmethod
    def fact(cls, *sources: SourceReference, confidence: float | None = None) -> Provenance:
        return cls(kind=AssertionKind.FACT, sources=list(sources), confidence=confidence)

    @classmethod
    def derived(
        cls,
        computation: str,
        *sources: SourceReference,
        confidence: float | None = None,
    ) -> Provenance:
        return cls(
            kind=AssertionKind.DERIVED,
            computation=computation,
            sources=list(sources),
            confidence=confidence,
        )

    @classmethod
    def estimate(
        cls,
        assumptions: dict[str, Any],
        *sources: SourceReference,
        confidence: float | None = None,
    ) -> Provenance:
        return cls(
            kind=AssertionKind.ESTIMATE,
            assumptions=assumptions,
            sources=list(sources),
            confidence=confidence,
        )

    @classmethod
    def generated(
        cls,
        model: str,
        *sources: SourceReference,
        confidence: float | None = None,
    ) -> Provenance:
        return cls(
            kind=AssertionKind.GENERATED,
            model=model,
            sources=list(sources),
            confidence=confidence,
        )


class Attributed(FrozenModel, Generic[T]):
    """A value bound to its provenance.

    Carrying the two together means a value cannot be moved between services,
    serialized into a report, or rendered in a UI while losing its attribution.
    """

    value: T
    provenance: Provenance
    label: str | None = None

    @property
    def is_model_authored(self) -> bool:
        return self.provenance.is_model_authored


__all__ = [
    "AssertionKind",
    "Attributed",
    "Provenance",
    "SourceLocator",
    "SourceReference",
    "SourceType",
]
