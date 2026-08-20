"""Numeric grounding — checking that the prose only quotes figures Atlas computed.

Phase 2 verified that a generated answer's *citations* resolved to real sources.
This is the same idea applied to the thing that actually matters in a financial
report: the numbers.

The rule being enforced is the ecosystem's central one. Claude may interpret,
compare, and explain the figures Atlas computed. It may not produce a figure of
its own. A model asked to "summarise the margin trend" will, given the chance,
write a plausible number rather than omit one — and a plausible number in a
research report is worse than no report.

**Rounding is permitted; invention is not.** A written figure is matched by
rounding each computed value to the number of significant digits the model
actually wrote, then comparing exactly. "$1.6 billion" therefore matches a
computed 1,613,590,000 — two significant digits, rounds to 1.6 billion — while
"$1.9 billion" does not match anything, at any tolerance. This is stricter and
more explainable than a percentage band, which either accepts a wrong number or
rejects a legitimately rounded one depending on magnitude.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

from pydantic import Field

from fie_schemas.base import FrozenModel

#: Written magnitudes and their multipliers.
_MAGNITUDES: dict[str, Decimal] = {
    "thousand": Decimal(1_000),
    "k": Decimal(1_000),
    "million": Decimal(1_000_000),
    "m": Decimal(1_000_000),
    "mm": Decimal(1_000_000),
    "billion": Decimal(1_000_000_000),
    "bn": Decimal(1_000_000_000),
    "b": Decimal(1_000_000_000),
    "trillion": Decimal(1_000_000_000_000),
    "tn": Decimal(1_000_000_000_000),
}

#: A number as a reader would write it, with optional currency symbol,
#: thousands separators, magnitude word, and percent or multiple suffix.
_NUMBER_PATTERN = re.compile(
    r"""
    (?P<currency>[$€£¥])?\s*
    (?P<sign>-|\()?\s*
    (?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    \)?
    \s*
    (?P<magnitude>thousand|million|billion|trillion|bn|tn|mm|k|m|b)?
    \s*
    (?P<suffix>%|x)?
    """,
    re.VERBOSE | re.IGNORECASE,
)

#: Four-digit integers in this range are read as calendar years rather than
#: financial quantities. Narrow and explicit, because every exemption here is a
#: hole in the check.
_YEAR_RANGE = range(1900, 2101)


class NumericKind(StrEnum):
    """What a written figure claims to be."""

    MONEY = "money"
    PERCENT = "percent"
    MULTIPLE = "multiple"
    PLAIN = "plain"
    #: A calendar year, exempt from grounding.
    YEAR = "year"


@dataclass(frozen=True)
class NumericClaim:
    """A figure as it appears in generated prose."""

    text: str
    value: Decimal
    kind: NumericKind
    position: int
    #: Significant digits the author actually wrote, which sets how precisely a
    #: computed value has to match.
    significant_digits: int

    @property
    def requires_grounding(self) -> bool:
        return self.kind is not NumericKind.YEAR


def _significant_digits(written: str) -> int:
    """Count the significant digits in a number as written."""
    digits = written.replace(",", "").replace("-", "").lstrip("0")
    if "." in digits:
        integer_part, _, fraction = digits.partition(".")
        if integer_part.strip("0") == "":
            # 0.0042 — leading zeros in the fraction are not significant.
            return len(fraction.lstrip("0")) or 1
        return len(integer_part) + len(fraction)
    stripped = digits.rstrip("0")
    return len(stripped) if stripped else 1


def _round_to_significant(value: Decimal, digits: int) -> Decimal:
    """Round to a number of significant digits."""
    if value == 0 or digits < 1:
        return Decimal(0)
    magnitude = value.copy_abs().adjusted()  # exponent of the leading digit
    exponent = Decimal(1).scaleb(magnitude - digits + 1)
    return (value / exponent).quantize(Decimal(1), rounding=ROUND_HALF_UP) * exponent


def extract_numeric_claims(text: str) -> list[NumericClaim]:
    """Find every figure a reader would take as a quantitative claim."""
    claims: list[NumericClaim] = []

    for match in _NUMBER_PATTERN.finditer(text):
        raw_number = match.group("number")
        if raw_number is None:
            continue

        written = raw_number.replace(",", "")
        try:
            value = Decimal(written)
        except ArithmeticError:  # pragma: no cover — the pattern guarantees a number
            continue

        magnitude = (match.group("magnitude") or "").lower()
        suffix = (match.group("suffix") or "").lower()
        currency = match.group("currency")
        negative = match.group("sign") in ("-", "(")

        # A bare four-digit integer with no unit is a year, not a quantity.
        is_year = (
            not magnitude
            and not suffix
            and not currency
            and "." not in written
            and int(value) in _YEAR_RANGE
        )

        if is_year:
            kind = NumericKind.YEAR
        elif suffix == "%":
            kind = NumericKind.PERCENT
        elif suffix == "x":
            kind = NumericKind.MULTIPLE
        elif currency or magnitude:
            kind = NumericKind.MONEY
        else:
            kind = NumericKind.PLAIN

        if magnitude:
            value = value * _MAGNITUDES[magnitude]
        if negative:
            value = -value

        claims.append(
            NumericClaim(
                text=match.group(0).strip(),
                value=value,
                kind=kind,
                position=match.start(),
                significant_digits=_significant_digits(written),
            )
        )

    return claims


class UnsupportedFigure(FrozenModel):
    """A figure in the prose that no computation produced."""

    text: str
    value: Decimal
    kind: NumericKind
    position: int


class NumericGroundingReport(FrozenModel):
    """The verdict on a generated narrative."""

    total_claims: int = Field(ge=0)
    grounded_claims: int = Field(ge=0)
    unsupported: list[UnsupportedFigure] = Field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        """True when every quantitative claim traces to a computed value."""
        return not self.unsupported

    @property
    def grounding_ratio(self) -> Decimal:
        if self.total_claims == 0:
            return Decimal(1)
        return (Decimal(self.grounded_claims) / Decimal(self.total_claims)).quantize(
            Decimal("0.0001")
        )


def check_numeric_grounding(text: str, allowed_values: set[Decimal]) -> NumericGroundingReport:
    """Verify every figure in ``text`` against the values Atlas computed.

    Args:
        allowed_values: Every magnitude Atlas produced — computed metrics and
            the reported figures they came from. A claim matches if rounding any
            allowed value to the claim's own precision reproduces it exactly.
    """
    claims = [claim for claim in extract_numeric_claims(text) if claim.requires_grounding]
    unsupported: list[UnsupportedFigure] = []

    for claim in claims:
        if not _is_supported(claim, allowed_values):
            unsupported.append(
                UnsupportedFigure(
                    text=claim.text,
                    value=claim.value,
                    kind=claim.kind,
                    position=claim.position,
                )
            )

    return NumericGroundingReport(
        total_claims=len(claims),
        grounded_claims=len(claims) - len(unsupported),
        unsupported=unsupported,
    )


def _is_supported(claim: NumericClaim, allowed_values: set[Decimal]) -> bool:
    """Whether some computed value rounds to what was written.

    Magnitudes are compared rather than signed values, because prose carries the
    sign in words rather than in the number: a reported net loss of -12,400,000
    is written "a loss of $12.4 million". Requiring the sign to match would
    reject correct writing, and this check exists to catch invented figures, not
    to police phrasing.
    """
    written = _round_to_significant(claim.value.copy_abs(), claim.significant_digits)

    for allowed in allowed_values:
        magnitude = allowed.copy_abs()
        if magnitude == claim.value.copy_abs():
            return True
        if _round_to_significant(magnitude, claim.significant_digits) == written:
            return True
        # A percentage may be written either as points ("23.4%") or, rarely, as
        # the underlying fraction. Both readings are checked so a correct
        # figure is not rejected over presentation.
        if claim.kind is NumericKind.PERCENT:
            as_fraction = magnitude * Decimal(100)
            if _round_to_significant(as_fraction, claim.significant_digits) == written:
                return True
    return False


def allowed_values_from(*value_groups: set[Decimal]) -> set[Decimal]:
    """Union of every source of legitimate figures, plus their negations.

    Negations are included because a narrative reasonably writes a reported loss
    as a positive magnitude ("a loss of $12.4 million") where the statement
    carries it as negative.
    """
    combined: set[Decimal] = set()
    for group in value_groups:
        combined |= group
    return combined | {-value for value in combined}


__all__ = [
    "NumericClaim",
    "NumericGroundingReport",
    "NumericKind",
    "UnsupportedFigure",
    "allowed_values_from",
    "check_numeric_grounding",
    "extract_numeric_claims",
]
