"""Discounted cash flow valuation.

Deterministic and reproducible: the same inputs always produce the same number,
and the inputs are recorded alongside it. A language model is not involved in
any step here — it is given the result afterwards and asked to explain it.

The output is an ESTIMATE rather than a DERIVED value, and the distinction is
not pedantic. A margin is arithmetic on reported facts. A DCF is a projection
resting on a discount rate and a terminal growth rate that a human chose, and
the shared provenance validator will not let one be constructed without stating
them. A valuation presented without its assumptions is an opinion wearing a
number's clothes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

from pydantic import Field

from atlas.analysis.results import Metric, Unit, estimated
from fie_common.errors import ValidationError
from fie_finance.money import Money, Rate, Shares
from fie_schemas.base import FrozenModel
from fie_schemas.provenance import SourceReference

_VERSION = "v1"
_COMPUTATION = f"atlas.dcf.{_VERSION}"

#: The discount rate must exceed terminal growth by at least this much. At
#: r == g the Gordon formula divides by zero; just above it, terminal value
#: explodes to a figure that is arithmetically valid and financially absurd.
MINIMUM_SPREAD = Decimal("0.005")

#: Present values are carried at this precision before final rounding.
_PRECISION = Decimal("0.0001")


@dataclass(frozen=True)
class DCFAssumptions:
    """The judgements a DCF rests on.

    Separated from the cash flows because these are the arguable part. A reader
    who disagrees with the valuation almost always disagrees with one of these
    numbers, so they are what the report has to surface.
    """

    discount_rate: Rate
    terminal_growth_rate: Rate
    #: Projection horizon in years. Beyond roughly ten years an explicit
    #: forecast is indistinguishable from the terminal value it feeds.
    projection_years: int = 5

    def __post_init__(self) -> None:
        if self.discount_rate.value <= 0:
            raise ValidationError(
                "the discount rate must be positive",
                details={"discount_rate": str(self.discount_rate.value)},
            )
        if self.projection_years < 1 or self.projection_years > 20:
            raise ValidationError(
                "the projection horizon must be between 1 and 20 years",
                details={"projection_years": self.projection_years},
            )

        spread = self.discount_rate.value - self.terminal_growth_rate.value
        if spread < MINIMUM_SPREAD:
            # The single most common way a DCF produces a nonsense answer.
            raise ValidationError(
                "the discount rate must exceed the terminal growth rate by at least "
                f"{MINIMUM_SPREAD}; otherwise terminal value is unbounded and the "
                "valuation is meaningless",
                details={
                    "discount_rate": str(self.discount_rate.value),
                    "terminal_growth_rate": str(self.terminal_growth_rate.value),
                    "spread": str(spread),
                },
            )

    def as_dict(self) -> dict[str, str]:
        return {
            "discount_rate": str(self.discount_rate.value),
            "terminal_growth_rate": str(self.terminal_growth_rate.value),
            "projection_years": str(self.projection_years),
        }


class ProjectedCashFlow(FrozenModel):
    """One year of the explicit forecast."""

    year: int = Field(ge=1)
    cash_flow: Money
    discount_factor: Decimal
    present_value: Money


class DCFValuation(FrozenModel):
    """A complete valuation, with every intermediate step retained.

    The per-year table is kept so a reviewer can reproduce the total by hand.
    An enterprise value nobody can check is a number, not an analysis.
    """

    entity_id: str
    currency: str
    projections: list[ProjectedCashFlow] = Field(default_factory=list)
    present_value_of_forecast: Money
    terminal_value: Money
    present_value_of_terminal: Money
    enterprise_value: Money
    equity_value: Money | None = None
    value_per_share: Money | None = None
    assumptions: dict[str, str] = Field(default_factory=dict)
    #: Share of enterprise value coming from the terminal value. Above roughly
    #: 75% the valuation is mostly an assumption about perpetuity.
    terminal_value_share: Decimal = Decimal(0)

    @property
    def is_terminal_dominated(self) -> bool:
        """Whether the terminal value carries most of the answer."""
        return self.terminal_value_share > Decimal("0.75")

    def to_metrics(self, sources: list[SourceReference] | None = None) -> list[Metric]:
        """Express the valuation as attributed metrics for a report."""
        metrics = [
            estimated(
                "enterprise_value",
                self.enterprise_value.amount,
                Unit.MONEY,
                assumptions=self.assumptions,
                sources=sources,
                currency=self.currency,
                inputs={"terminal_value_share": self.terminal_value_share},
            )
        ]
        if self.equity_value is not None:
            metrics.append(
                estimated(
                    "equity_value",
                    self.equity_value.amount,
                    Unit.MONEY,
                    assumptions=self.assumptions,
                    sources=sources,
                    currency=self.currency,
                )
            )
        if self.value_per_share is not None:
            metrics.append(
                estimated(
                    "value_per_share",
                    self.value_per_share.amount,
                    Unit.MONEY,
                    assumptions=self.assumptions,
                    sources=sources,
                    currency=self.currency,
                )
            )
        return metrics


@dataclass
class DCFModel:
    """Runs a discounted cash flow.

    Cash flows are supplied rather than forecast here: projecting them is a
    judgement, and burying it inside the arithmetic would hide the part of the
    valuation most worth arguing about.
    """

    assumptions: DCFAssumptions
    sources: list[SourceReference] = field(default_factory=list)

    def value(
        self,
        entity_id: str,
        cash_flows: list[Money],
        *,
        net_debt: Money | None = None,
        diluted_shares: Shares | None = None,
    ) -> DCFValuation:
        """Value a stream of projected free cash flows.

        Args:
            cash_flows: Projected free cash flow per year, first year first.
            net_debt: Subtracted from enterprise value to reach equity value.
                Omitted when the capital structure is unknown, in which case
                only an enterprise value is returned.

        Raises:
            ValidationError: if the cash flows are empty, mix currencies, or do
                not match the assumed projection horizon.
        """
        if not cash_flows:
            raise ValidationError("a DCF requires at least one projected cash flow")
        if len(cash_flows) != self.assumptions.projection_years:
            raise ValidationError(
                "the number of projected cash flows must match the projection horizon",
                details={
                    "cash_flows": len(cash_flows),
                    "projection_years": self.assumptions.projection_years,
                },
            )

        currency = cash_flows[0].currency
        if any(flow.currency != currency for flow in cash_flows):
            raise ValidationError("every projected cash flow must be in one currency")

        rate = self.assumptions.discount_rate.value
        growth = self.assumptions.terminal_growth_rate.value

        projections: list[ProjectedCashFlow] = []
        present_value = Money.zero(currency)

        for index, flow in enumerate(cash_flows, start=1):
            discount_factor = (Decimal(1) / ((Decimal(1) + rate) ** index)).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
            discounted = flow.scaled_by(discount_factor)
            present_value = present_value + discounted
            projections.append(
                ProjectedCashFlow(
                    year=index,
                    cash_flow=flow,
                    discount_factor=discount_factor,
                    present_value=discounted,
                )
            )

        # Gordon growth: the final year's cash flow grown one more period, then
        # capitalised. `__post_init__` has already guaranteed rate > growth.
        final_flow = cash_flows[-1]
        terminal_value = final_flow.scaled_by(
            ((Decimal(1) + growth) / (rate - growth)).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        )
        terminal_discount = (
            Decimal(1) / ((Decimal(1) + rate) ** self.assumptions.projection_years)
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        present_value_of_terminal = terminal_value.scaled_by(terminal_discount)

        enterprise_value = present_value + present_value_of_terminal

        equity_value = None
        value_per_share = None
        if net_debt is not None:
            if net_debt.currency != currency:
                raise ValidationError("net debt must be in the same currency as the cash flows")
            equity_value = enterprise_value - net_debt
            if diluted_shares is not None:
                value_per_share = equity_value.divided_by(diluted_shares.count)

        terminal_share = (
            present_value_of_terminal.ratio_to(enterprise_value)
            if not enterprise_value.is_zero
            else Decimal(0)
        )

        return DCFValuation(
            entity_id=entity_id,
            currency=currency,
            projections=projections,
            present_value_of_forecast=present_value.rounded(),
            terminal_value=terminal_value.rounded(),
            present_value_of_terminal=present_value_of_terminal.rounded(),
            enterprise_value=enterprise_value.rounded(),
            equity_value=equity_value.rounded() if equity_value else None,
            value_per_share=value_per_share.rounded(Decimal("0.0001")) if value_per_share else None,
            assumptions=self.assumptions.as_dict(),
            terminal_value_share=terminal_share,
        )


def project_cash_flows(base_cash_flow: Money, growth_rate: Rate, years: int) -> list[Money]:
    """Grow a base cash flow at a constant rate.

    The simplest possible projection, offered explicitly so that a caller using
    it is visibly choosing it. Real forecasts vary the rate by year, and this
    function existing under a plain name keeps that choice from looking like a
    default.

    Raises:
        ValidationError: if the horizon is not positive.
    """
    if years < 1:
        raise ValidationError("a projection must cover at least one year")

    flows: list[Money] = []
    current = base_cash_flow
    for _ in range(years):
        current = current.scaled_by(Decimal(1) + growth_rate.value)
        flows.append(current)
    return flows


def sensitivity_grid(
    entity_id: str,
    cash_flows: list[Money],
    *,
    discount_rates: list[Rate],
    terminal_growth_rates: list[Rate],
    projection_years: int,
    net_debt: Money | None = None,
    diluted_shares: Shares | None = None,
) -> dict[str, dict[str, str]]:
    """Value across a grid of assumptions.

    A single DCF number invites false precision. The grid shows how much of the
    answer is the assumption rather than the business, which is the honest way
    to present a projection — and it makes an implausible combination visible
    instead of merely producing a large number.

    Combinations that violate the rate-versus-growth constraint are reported as
    ``"n/a"`` rather than silently skipped.
    """
    grid: dict[str, dict[str, str]] = {}
    for rate in discount_rates:
        row: dict[str, str] = {}
        for growth in terminal_growth_rates:
            try:
                assumptions = DCFAssumptions(
                    discount_rate=rate,
                    terminal_growth_rate=growth,
                    projection_years=projection_years,
                )
            except ValidationError:
                row[str(growth)] = "n/a"
                continue

            valuation = DCFModel(assumptions).value(
                entity_id, cash_flows, net_debt=net_debt, diluted_shares=diluted_shares
            )
            reported = (
                valuation.value_per_share or valuation.equity_value or (valuation.enterprise_value)
            )
            row[str(growth)] = str(reported.amount)
        grid[str(rate)] = row
    return grid


def weighted_average_cost_of_capital(
    *,
    equity_value: Money,
    debt_value: Money,
    cost_of_equity: Rate,
    cost_of_debt: Rate,
    tax_rate: Rate,
) -> Rate:
    """WACC from capital structure and component costs.

    Provided so a discount rate can be computed from stated inputs rather than
    asserted. The tax shield on debt is applied because interest is deductible;
    omitting it overstates the cost of capital and understates every valuation
    that uses it.

    Raises:
        ValidationError: if total capital is zero.
    """
    total = equity_value + debt_value
    if total.is_zero:
        raise ValidationError("cannot compute WACC with zero total capital")

    equity_weight = equity_value.ratio_to(total)
    debt_weight = debt_value.ratio_to(total)
    after_tax_debt_cost = cost_of_debt.value * (Decimal(1) - tax_rate.value)

    return Rate.of(
        (equity_weight * cost_of_equity.value + debt_weight * after_tax_debt_cost).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_UP
        )
    )


__all__ = [
    "MINIMUM_SPREAD",
    "DCFAssumptions",
    "DCFModel",
    "DCFValuation",
    "ProjectedCashFlow",
    "project_cash_flows",
    "sensitivity_grid",
    "weighted_average_cost_of_capital",
]
