"""Fiscal periods.

A financial figure without its period is not a fact. "Revenue was $412.6m" is
only true of a stated span, and comparing a quarter against a year is the most
common way an analysis produces a confidently wrong growth rate — so periods
carry their kind and refuse comparisons that do not make sense.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import Field, model_validator

from fie_schemas.base import FrozenModel


class PeriodKind(StrEnum):
    """The span a figure covers."""

    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    #: Cumulative from the start of the fiscal year.
    YEAR_TO_DATE = "year_to_date"
    #: A balance sheet is a position at an instant, not a flow over a span.
    INSTANT = "instant"

    @property
    def is_flow(self) -> bool:
        """Whether figures in this period accumulate over time.

        Income and cash flow items are flows; balance sheet items are not.
        Summing four balance sheets is meaningless, and this is what lets the
        statement models say so.
        """
        return self is not PeriodKind.INSTANT


class FiscalPeriod(FrozenModel):
    """A dated reporting period.

    Fiscal years frequently do not align with calendar years, so the label and
    the dates are stored separately rather than deriving one from the other.
    """

    kind: PeriodKind
    fiscal_year: int = Field(ge=1900, le=2200)
    #: 1-4 for quarterly periods, ``None`` for annual and instant.
    fiscal_quarter: int | None = Field(default=None, ge=1, le=4)
    start_date: date | None = None
    end_date: date

    @model_validator(mode="after")
    def _validate(self) -> FiscalPeriod:
        if self.kind is PeriodKind.QUARTERLY and self.fiscal_quarter is None:
            raise ValueError("a quarterly period must state which quarter")
        if self.kind in (PeriodKind.ANNUAL, PeriodKind.INSTANT) and self.fiscal_quarter:
            raise ValueError(f"a {self.kind} period must not carry a quarter")

        if self.kind is PeriodKind.INSTANT:
            if self.start_date is not None and self.start_date != self.end_date:
                raise ValueError("an instant period is a point in time; it has no span")
        elif self.start_date is not None and self.start_date >= self.end_date:
            raise ValueError("start_date must precede end_date")
        return self

    @classmethod
    def annual(
        cls, fiscal_year: int, end_date: date, start_date: date | None = None
    ) -> FiscalPeriod:
        return cls(
            kind=PeriodKind.ANNUAL,
            fiscal_year=fiscal_year,
            end_date=end_date,
            start_date=start_date,
        )

    @classmethod
    def quarterly(
        cls, fiscal_year: int, quarter: int, end_date: date, start_date: date | None = None
    ) -> FiscalPeriod:
        return cls(
            kind=PeriodKind.QUARTERLY,
            fiscal_year=fiscal_year,
            fiscal_quarter=quarter,
            end_date=end_date,
            start_date=start_date,
        )

    @classmethod
    def instant(cls, fiscal_year: int, as_of: date) -> FiscalPeriod:
        return cls(kind=PeriodKind.INSTANT, fiscal_year=fiscal_year, end_date=as_of)

    @property
    def label(self) -> str:
        """Human-readable identifier, e.g. ``FY2025`` or ``Q3 FY2025``."""
        if self.kind is PeriodKind.QUARTERLY:
            return f"Q{self.fiscal_quarter} FY{self.fiscal_year}"
        if self.kind is PeriodKind.INSTANT:
            return f"As of {self.end_date.isoformat()}"
        if self.kind is PeriodKind.YEAR_TO_DATE:
            return f"FY{self.fiscal_year} YTD"
        return f"FY{self.fiscal_year}"

    def is_comparable_to(self, other: FiscalPeriod) -> bool:
        """Whether a growth rate between these two periods is meaningful.

        Comparing a quarter to a full year produces a number that looks like
        growth and is not, which is why every growth calculation checks this
        before it will run.
        """
        if self.kind is not other.kind:
            return False
        if self.kind is PeriodKind.QUARTERLY:
            return self.fiscal_quarter == other.fiscal_quarter
        return True

    def __lt__(self, other: FiscalPeriod) -> bool:
        return self.end_date < other.end_date


__all__ = ["FiscalPeriod", "PeriodKind"]
