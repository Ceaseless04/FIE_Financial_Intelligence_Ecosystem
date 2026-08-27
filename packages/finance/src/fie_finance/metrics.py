"""The result type every deterministic calculation returns.

This module is where the ecosystem's central rule stops being prose:

> Claude is the primary reasoning model. It is never the source of truth for a
> financial calculation.

A :class:`Metric` is built through :func:`derived`, which constructs
``Provenance.derived(computation=...)``. The shared provenance validator refuses
a DERIVED assertion that names a model, so a number produced here *cannot* be
attributed to a language model — not by convention, but because the object will
not construct.

Every metric also carries the inputs it was computed from. A valuation nobody
can reproduce is a valuation nobody should act on, and recording the inputs
turns "trust the number" into "here is the arithmetic".
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator

from fie_common.errors import ValidationError
from fie_finance.money import Money, decimal_field, to_decimal
from fie_schemas.base import FrozenModel
from fie_schemas.provenance import Provenance, SourceReference


class Unit(StrEnum):
    """What a metric's magnitude means.

    Carried explicitly because ``0.42`` as a ratio, as 42 percent, and as
    forty-two cents are three different claims, and a report that confuses them
    is wrong in a way that reads perfectly well.
    """

    RATIO = "ratio"
    PERCENT = "percent"
    MONEY = "money"
    #: A multiple, as in "12.4x EBITDA".
    TIMES = "times"
    DAYS = "days"
    YEARS = "years"
    SHARES = "shares"


class Metric(FrozenModel):
    """A single computed figure, bound to how it was produced."""

    name: str = Field(min_length=1)
    value: Decimal
    unit: Unit
    #: Set when ``unit`` is MONEY; meaningless otherwise.
    currency: str | None = None
    #: The named inputs, rendered as strings so the record survives transport
    #: and stays readable in a report appendix.
    inputs: dict[str, str] = Field(default_factory=dict)
    provenance: Provenance
    #: Present when the figure could not be computed from what was available.
    unavailable_reason: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _reject_float(cls, value: Any) -> Decimal:
        return decimal_field(value)

    @property
    def is_available(self) -> bool:
        return self.unavailable_reason is None

    @property
    def as_money(self) -> Money | None:
        if self.unit is not Unit.MONEY or self.currency is None:
            return None
        return Money(amount=self.value, currency=self.currency)

    def rendered(self, places: int = 2) -> str:
        """Format for display, with the unit made explicit."""
        exponent = Decimal(1).scaleb(-places)
        quantized = self.value.quantize(exponent, rounding=ROUND_HALF_UP)
        if self.unit is Unit.PERCENT:
            return f"{quantized}%"
        if self.unit is Unit.TIMES:
            return f"{quantized}x"
        if self.unit is Unit.MONEY:
            return f"{quantized} {self.currency or ''}".strip()
        if self.unit in (Unit.DAYS, Unit.YEARS):
            return f"{quantized} {self.unit}"
        return str(quantized)


def derived(
    name: str,
    value: Decimal | int | str,
    unit: Unit,
    *,
    computation: str,
    sources: list[SourceReference] | None = None,
    inputs: dict[str, Any] | None = None,
    currency: str | None = None,
) -> Metric:
    """Build a metric produced by a deterministic computation.

    Args:
        computation: Stable identifier of the routine and its version, e.g.
            ``atlas.ratios.net_margin.v1``. Versioned because changing a formula
            changes historical results, and a reader needs to know which one
            produced the number in front of them.
    """
    return Metric(
        name=name,
        value=to_decimal(value),
        unit=unit,
        currency=currency,
        inputs={key: str(item) for key, item in (inputs or {}).items()},
        provenance=Provenance.derived(computation, *(sources or [])),
    )


def estimated(
    name: str,
    value: Decimal | int | str,
    unit: Unit,
    *,
    assumptions: dict[str, Any],
    sources: list[SourceReference] | None = None,
    inputs: dict[str, Any] | None = None,
    currency: str | None = None,
) -> Metric:
    """Build a metric that rests on assumptions rather than only on facts.

    A discounted cash flow is not derived from a filing the way a margin is; it
    is a projection resting on a discount rate and a terminal growth rate that
    someone chose. The shared validator requires those assumptions to be stated,
    which is what makes the resulting number arguable instead of authoritative.

    Raises:
        ValidationError: if no assumptions are supplied.
    """
    if not assumptions:
        raise ValidationError(
            "an estimate must state the assumptions it rests on",
            details={"metric": name},
        )
    return Metric(
        name=name,
        value=to_decimal(value),
        unit=unit,
        currency=currency,
        inputs={key: str(item) for key, item in (inputs or {}).items()},
        provenance=Provenance.estimate(
            {key: str(item) for key, item in assumptions.items()}, *(sources or [])
        ),
    )


def unavailable(name: str, unit: Unit, *, reason: str, computation: str) -> Metric:
    """A metric that could not be computed.

    Returned rather than raising, and rather than substituting zero. A missing
    input is a fact about the filing, and a report that says "not disclosed" is
    more useful — and far safer — than one showing a zero that reads as a real
    measurement.
    """
    return Metric(
        name=name,
        value=Decimal(0),
        unit=unit,
        inputs={},
        provenance=Provenance.derived(computation),
        unavailable_reason=reason,
    )


class AnalysisResult(FrozenModel):
    """A named group of metrics produced by one analysis pass."""

    entity_id: str = Field(min_length=1)
    period_label: str
    metrics: list[Metric] = Field(default_factory=list)

    def get(self, name: str) -> Metric | None:
        return next((metric for metric in self.metrics if metric.name == name), None)

    @property
    def available(self) -> list[Metric]:
        return [metric for metric in self.metrics if metric.is_available]

    @property
    def unavailable(self) -> list[Metric]:
        return [metric for metric in self.metrics if not metric.is_available]

    def numeric_values(self) -> set[Decimal]:
        """Every computed magnitude, for verifying a narrative against them."""
        return {metric.value for metric in self.available}


__all__ = [
    "AnalysisResult",
    "Metric",
    "Unit",
    "derived",
    "estimated",
    "unavailable",
]
