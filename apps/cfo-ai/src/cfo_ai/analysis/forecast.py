"""Forecasting, deterministically.

Four methods, all arithmetic, none of them clever. That is deliberate: a
forecast a CFO cannot reproduce on a whiteboard is a forecast they cannot
defend in a board meeting, and every method here can be checked by hand from
the history and the stated window.

Every result is an **ESTIMATE**, never a DERIVED value. The shared provenance
validator refuses an estimate that does not state its assumptions, so a
projection physically cannot travel without the method and window that produced
it. The distinction matters more here than anywhere else in the ecosystem: a
variance is a fact about the past, and a forecast is a claim about the future
wearing the same units.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

from pydantic import Field

from fie_common.errors import ValidationError
from fie_finance.metrics import Metric, Unit, estimated
from fie_finance.money import Money
from fie_finance.periods import FiscalPeriod
from fie_schemas.base import FrozenModel
from fie_schemas.provenance import SourceReference

COMPUTATION = "cfo_ai.forecast.v1"

_MONEY_EXPONENT = Decimal("0.01")


class ForecastMethod(StrEnum):
    """How a projection was produced."""

    #: The most recent period, repeated. Honest about assuming nothing changes.
    RUN_RATE = "run_rate"
    #: Mean of the last ``window`` periods. Damps a single unusual month.
    MOVING_AVERAGE = "moving_average"
    #: Least-squares straight line through the history, extended forward.
    LINEAR_TREND = "linear_trend"
    #: The same period one year earlier, which is what a seasonal business
    #: actually plans against.
    SEASONAL_NAIVE = "seasonal_naive"

    @property
    def minimum_history(self) -> int:
        """Periods required before the method means anything."""
        return {
            ForecastMethod.RUN_RATE: 1,
            ForecastMethod.MOVING_AVERAGE: 2,
            ForecastMethod.LINEAR_TREND: 3,
            ForecastMethod.SEASONAL_NAIVE: 12,
        }[self]


class Observation(FrozenModel):
    """One historical period's actual figure."""

    period: FiscalPeriod
    amount: Money


class ForecastPoint(FrozenModel):
    """One projected period."""

    #: 1 for the first period beyond the history.
    step: int = Field(ge=1)
    amount: Money


class Forecast(FrozenModel):
    """A projection and everything needed to argue with it."""

    account_code: str
    method: ForecastMethod
    currency: str
    points: list[ForecastPoint]
    #: How many historical periods the method actually consumed.
    history_used: int = Field(ge=1)
    #: Stated inputs. Carried into ESTIMATE provenance verbatim.
    assumptions: dict[str, str] = Field(default_factory=dict)

    @property
    def total(self) -> Money:
        total = Money.zero(self.currency)
        for point in self.points:
            total = total + point.amount
        return total

    def to_metrics(self, sources: list[SourceReference] | None = None) -> list[Metric]:
        """One ESTIMATE per projected period, plus the total.

        Never DERIVED. A projection presented as a computed fact is the failure
        this whole ecosystem is arranged to prevent.
        """
        metrics = [
            estimated(
                f"forecast.{self.account_code}.t{point.step}",
                point.amount.amount,
                Unit.MONEY,
                assumptions=self.assumptions,
                currency=self.currency,
                sources=sources,
                inputs={"step": str(point.step), "method": str(self.method)},
            )
            for point in self.points
        ]
        metrics.append(
            estimated(
                f"forecast.{self.account_code}.total",
                self.total.amount,
                Unit.MONEY,
                assumptions=self.assumptions,
                currency=self.currency,
                sources=sources,
                inputs={"periods": str(len(self.points)), "method": str(self.method)},
            )
        )
        return metrics


def _validated_history(history: list[Observation], method: ForecastMethod) -> list[Observation]:
    """Order the history and refuse to forecast from too little of it."""
    if not history:
        raise ValidationError("a forecast needs history", details={"method": str(method)})

    currencies = {observation.amount.currency for observation in history}
    if len(currencies) > 1:
        raise ValidationError(
            "cannot forecast across mixed currencies",
            details={"currencies": sorted(currencies)},
        )

    ordered = sorted(history, key=lambda observation: observation.period.end_date)

    seen: set[str] = set()
    for observation in ordered:
        if observation.period.key in seen:
            raise ValidationError(
                "the history contains two observations for one period, which "
                "would double-count into every projection built from it",
                details={"period": observation.period.label},
            )
        seen.add(observation.period.key)

    if len(ordered) < method.minimum_history:
        raise ValidationError(
            f"{method} needs at least {method.minimum_history} periods of history "
            f"to mean anything; {len(ordered)} supplied",
            details={"method": str(method), "supplied": len(ordered)},
        )
    return ordered


def forecast(
    account_code: str,
    history: list[Observation],
    *,
    periods: int = 3,
    method: ForecastMethod = ForecastMethod.RUN_RATE,
    window: int | None = None,
) -> Forecast:
    """Project an account forward.

    Args:
        window: Periods the method looks back over. Defaults to the whole
            history for a moving average and a trend. Stated in the assumptions
            either way, because "the average" means nothing without it.

    Raises:
        ValidationError: on empty or too-short history, mixed currencies,
            duplicate periods, or a non-positive horizon.
    """
    if periods < 1:
        raise ValidationError("a forecast must project at least one period")

    ordered = _validated_history(history, method)
    currency = ordered[0].amount.currency
    effective_window = min(window or len(ordered), len(ordered))

    if method is ForecastMethod.RUN_RATE:
        value = ordered[-1].amount
        amounts = [value] * periods
        used = 1
        assumptions = {
            "method": "run rate",
            "basis": f"the {ordered[-1].period.label} actual, repeated",
            "implies": "no growth, no seasonality, no change in run rate",
        }

    elif method is ForecastMethod.MOVING_AVERAGE:
        recent = ordered[-effective_window:]
        total = Money.zero(currency)
        for observation in recent:
            total = total + observation.amount
        value = total.divided_by(len(recent))
        amounts = [value] * periods
        used = len(recent)
        assumptions = {
            "method": "moving average",
            "window": f"{len(recent)} periods",
            "implies": "the recent average continues; a trend within the window is flattened",
        }

    elif method is ForecastMethod.LINEAR_TREND:
        recent = ordered[-effective_window:]
        slope, intercept = _least_squares(recent)
        amounts = []
        for step in range(1, periods + 1):
            index = Decimal(len(recent) - 1 + step)
            projected = (intercept + slope * index).quantize(
                _MONEY_EXPONENT, rounding=ROUND_HALF_UP
            )
            amounts.append(Money(amount=projected, currency=currency))
        used = len(recent)
        assumptions = {
            "method": "least-squares linear trend",
            "window": f"{len(recent)} periods",
            "slope_per_period": str(slope.quantize(_MONEY_EXPONENT, rounding=ROUND_HALF_UP)),
            "implies": "the observed straight line continues; it does not bend or saturate",
        }

    else:  # SEASONAL_NAIVE
        amounts = []
        for step in range(1, periods + 1):
            # The same period one year back: 12 behind the point being
            # projected, which is `len - 12 + step` into the ordered history.
            offset = len(ordered) - 12 + (step - 1)
            if offset < 0 or offset >= len(ordered):
                raise ValidationError(
                    "the history does not reach back a full year for every projected period",
                    details={"periods": periods, "history": len(ordered)},
                )
            amounts.append(ordered[offset].amount)
        used = 12
        assumptions = {
            "method": "seasonal naive",
            "basis": "the same period one year earlier",
            "implies": "last year's shape repeats; it carries no growth",
        }

    return Forecast(
        account_code=account_code.strip().upper(),
        method=method,
        currency=currency,
        points=[
            ForecastPoint(step=step, amount=amount) for step, amount in enumerate(amounts, start=1)
        ],
        history_used=used,
        assumptions=assumptions,
    )


def _least_squares(observations: list[Observation]) -> tuple[Decimal, Decimal]:
    """Slope and intercept of the best-fit line, in Decimal throughout.

    Indices are 0..n-1, so the intercept is the fitted value at the first
    observation. Kept in Decimal rather than handed to a float statistics
    routine, because a slope computed in binary floating point reintroduces
    exactly the error every other module here refuses.
    """
    count = Decimal(len(observations))
    indices = [Decimal(index) for index in range(len(observations))]
    values = [observation.amount.amount for observation in observations]

    sum_x = sum(indices, Decimal(0))
    sum_y = sum(values, Decimal(0))
    sum_xy = sum((x * y for x, y in zip(indices, values, strict=True)), Decimal(0))
    sum_xx = sum((x * x for x in indices), Decimal(0))

    denominator = count * sum_xx - sum_x * sum_x
    if denominator == 0:  # pragma: no cover — needs >= 3 distinct indices
        raise ValidationError("cannot fit a trend to a single distinct period")

    slope = (count * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / count
    return slope, intercept


__all__ = [
    "COMPUTATION",
    "Forecast",
    "ForecastMethod",
    "ForecastPoint",
    "Observation",
    "forecast",
]
