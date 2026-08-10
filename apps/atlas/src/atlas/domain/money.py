"""Monetary and rate primitives.

Everything financial in Atlas is built on ``Decimal``. Binary floating point is
not merely imprecise here, it is wrong in a way that compounds: ``0.1 + 0.2``
is not ``0.3``, and a discounted cash flow multiplies that error across ten
periods before anyone reads the output. A valuation a reviewer cannot reproduce
by hand is not a valuation.

Three rules are enforced rather than documented:

1. **No float ever enters.** A ``float`` is rejected at construction, because
   ``Decimal(0.1)`` silently becomes ``0.1000000000000000055511151231257827``.
   Callers pass a string, an int, or a ``Decimal``.
2. **Currencies never mix silently.** Adding USD to EUR raises rather than
   producing a number that looks plausible and means nothing.
3. **Rounding is explicit and deliberate.** Internal arithmetic keeps full
   precision; rounding happens where a value is presented or reported, using
   banker's-rounding-free ``ROUND_HALF_UP`` because that is what a financial
   reader expects and what a spreadsheet does.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from pydantic import Field, field_validator, model_validator

from fie_schemas.base import FrozenModel

#: ISO 4217 alphabetic code.
_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")

#: Precision carried through intermediate arithmetic. Wide enough that rounding
#: never bites before the final presentation step.
_INTERNAL_EXPONENT = Decimal("0.000001")

#: What a financial statement is normally presented to.
CENTS = Decimal("0.01")


def to_decimal(value: Decimal | int | str) -> Decimal:
    """Convert to ``Decimal``, refusing the lossy paths.

    Raises:
        TypeError: if given a ``float``. The conversion is silent and lossy, so
            it is rejected at the boundary rather than producing a number that
            is almost right.
        ValueError: if the value is not a well-formed number.
    """
    if isinstance(value, bool):
        raise TypeError("a boolean is not a monetary amount")
    if isinstance(value, float):
        raise TypeError(
            "float is not accepted for financial values because the conversion "
            "is lossy; pass a string, an int, or a Decimal"
        )
    if isinstance(value, Decimal):
        candidate = value
    else:
        try:
            candidate = Decimal(str(value))
        except InvalidOperation as error:
            raise ValueError(f"{value!r} is not a valid decimal number") from error

    if not candidate.is_finite():
        raise ValueError("financial values must be finite")
    return candidate


def decimal_field(value: Any) -> Decimal:
    """``to_decimal`` for use inside a Pydantic validator.

    Pydantic converts only ``ValueError`` and ``AssertionError`` into a
    ``ValidationError``; a ``TypeError`` propagates untouched. Since JSON
    numbers deserialize to ``float``, a client posting ``{"amount": 1234.56}``
    would otherwise produce a 500 at the API edge instead of a 422. The value is
    still rejected — the conversion is lossy — but it is rejected as bad input.
    """
    try:
        return to_decimal(value)
    except TypeError as error:
        raise ValueError(str(error)) from error


class Money(FrozenModel):
    """An amount in a single currency.

    Immutable, so a value handed to a calculation cannot be mutated behind the
    caller's back and every arithmetic operation returns a new instance.
    """

    amount: Decimal
    currency: str = Field(default="USD", min_length=3, max_length=3)

    @field_validator("amount", mode="before")
    @classmethod
    def _reject_float(cls, value: Any) -> Decimal:
        return decimal_field(value)

    @field_validator("currency")
    @classmethod
    def _validate_currency(cls, value: str) -> str:
        upper = value.upper()
        if not _CURRENCY_PATTERN.match(upper):
            raise ValueError(f"{value!r} is not an ISO 4217 alphabetic currency code")
        return upper

    # -- construction --------------------------------------------------------

    @classmethod
    def of(cls, amount: Decimal | int | str, currency: str = "USD") -> Money:
        return cls(amount=to_decimal(amount), currency=currency)

    @classmethod
    def zero(cls, currency: str = "USD") -> Money:
        return cls(amount=Decimal(0), currency=currency)

    @classmethod
    def millions(cls, amount: Decimal | int | str, currency: str = "USD") -> Money:
        """Filings report in millions; this keeps the unit conversion in one place."""
        return cls(amount=to_decimal(amount) * Decimal(1_000_000), currency=currency)

    @classmethod
    def thousands(cls, amount: Decimal | int | str, currency: str = "USD") -> Money:
        return cls(amount=to_decimal(amount) * Decimal(1_000), currency=currency)

    # -- arithmetic ----------------------------------------------------------

    def _require_same_currency(self, other: Money) -> None:
        if self.currency != other.currency:
            raise ValueError(
                f"cannot combine {self.currency} and {other.currency}; "
                "convert explicitly with a stated exchange rate"
            )

    def __add__(self, other: Money) -> Money:
        self._require_same_currency(other)
        return Money(amount=self.amount + other.amount, currency=self.currency)

    def __sub__(self, other: Money) -> Money:
        self._require_same_currency(other)
        return Money(amount=self.amount - other.amount, currency=self.currency)

    def __neg__(self) -> Money:
        return Money(amount=-self.amount, currency=self.currency)

    def __abs__(self) -> Money:
        return Money(amount=abs(self.amount), currency=self.currency)

    def scaled_by(self, factor: Decimal | int | str) -> Money:
        """Multiply by a dimensionless factor (a growth rate, a share count)."""
        return Money(amount=self.amount * to_decimal(factor), currency=self.currency)

    def divided_by(self, divisor: Decimal | int | str) -> Money:
        """Divide by a dimensionless divisor.

        Raises:
            ZeroDivisionError: rather than returning infinity, which would
                propagate into a report as a plausible-looking number.
        """
        value = to_decimal(divisor)
        if value == 0:
            raise ZeroDivisionError("cannot divide a monetary amount by zero")
        return Money(
            amount=(self.amount / value).quantize(_INTERNAL_EXPONENT, rounding=ROUND_HALF_UP),
            currency=self.currency,
        )

    def ratio_to(self, other: Money) -> Decimal:
        """Dimensionless ratio between two amounts in the same currency.

        Raises:
            ZeroDivisionError: if the denominator is zero.
        """
        self._require_same_currency(other)
        if other.amount == 0:
            raise ZeroDivisionError("cannot compute a ratio against zero")
        return (self.amount / other.amount).quantize(_INTERNAL_EXPONENT, rounding=ROUND_HALF_UP)

    # -- presentation --------------------------------------------------------

    def rounded(self, exponent: Decimal = CENTS) -> Money:
        """Round for presentation. Never called mid-calculation."""
        return Money(
            amount=self.amount.quantize(exponent, rounding=ROUND_HALF_UP),
            currency=self.currency,
        )

    @property
    def is_zero(self) -> bool:
        return self.amount == 0

    @property
    def is_negative(self) -> bool:
        return self.amount < 0

    def __str__(self) -> str:
        return f"{self.amount.quantize(CENTS, rounding=ROUND_HALF_UP)} {self.currency}"

    def __lt__(self, other: Money) -> bool:
        self._require_same_currency(other)
        return self.amount < other.amount

    def __le__(self, other: Money) -> bool:
        self._require_same_currency(other)
        return self.amount <= other.amount

    def __gt__(self, other: Money) -> bool:
        self._require_same_currency(other)
        return self.amount > other.amount

    def __ge__(self, other: Money) -> bool:
        self._require_same_currency(other)
        return self.amount >= other.amount


def sum_money(amounts: list[Money], *, currency: str = "USD") -> Money:
    """Total a list, refusing to add across currencies.

    An explicit currency is required for the empty case: returning a bare zero
    with a guessed currency is how a EUR statement silently totals to USD.
    """
    total = Money.zero(currency)
    for amount in amounts:
        total = total + amount
    return total


class Rate(FrozenModel):
    """A dimensionless rate — a discount rate, a growth rate, a margin.

    Stored as a decimal fraction (``0.085`` for 8.5%) rather than as percentage
    points, because every formula that consumes one wants the fraction and
    converting at each call site is where the factor-of-100 errors live.
    """

    value: Decimal

    @field_validator("value", mode="before")
    @classmethod
    def _reject_float(cls, value: Any) -> Decimal:
        return decimal_field(value)

    @classmethod
    def of(cls, value: Decimal | int | str) -> Rate:
        return cls(value=to_decimal(value))

    @classmethod
    def from_percent(cls, percent: Decimal | int | str) -> Rate:
        """Build from percentage points: ``from_percent("8.5")`` is 0.085."""
        return cls(value=to_decimal(percent) / Decimal(100))

    @property
    def as_percent(self) -> Decimal:
        return (self.value * Decimal(100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def __str__(self) -> str:
        return f"{self.as_percent}%"


class Shares(FrozenModel):
    """A share count.

    Distinct from a plain int so that per-share arithmetic cannot accidentally
    divide by a dollar amount, and so a fractional count from a weighted
    diluted average keeps its precision.
    """

    count: Decimal

    @field_validator("count", mode="before")
    @classmethod
    def _reject_float(cls, value: Any) -> Decimal:
        return decimal_field(value)

    @model_validator(mode="after")
    def _must_be_positive(self) -> Shares:
        if self.count <= 0:
            raise ValueError("a share count must be positive")
        return self

    @classmethod
    def of(cls, count: Decimal | int | str) -> Shares:
        return cls(count=to_decimal(count))

    @classmethod
    def millions(cls, count: Decimal | int | str) -> Shares:
        return cls(count=to_decimal(count) * Decimal(1_000_000))


__all__ = [
    "CENTS",
    "Money",
    "Rate",
    "Shares",
    "decimal_field",
    "sum_money",
    "to_decimal",
]
