"""Relative valuation against a peer set.

Multiples are the sanity check on a DCF: two methods disagreeing by an order of
magnitude means one of them is wrong, and finding out which is the analysis.

The peer set is supplied by the caller — typically from MarketMind's
``COMPETES_WITH`` edges — rather than chosen here. Peer selection is a judgement
that changes the answer, so it is recorded as an assumption rather than buried
in a heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from pydantic import Field

from atlas.analysis.results import Metric, Unit, derived, estimated, unavailable
from atlas.domain.money import Money, Shares
from fie_common.errors import ValidationError
from fie_schemas.base import FrozenModel

_VERSION = "v1"
_PRECISION = Decimal("0.0001")


class PeerMultiple(FrozenModel):
    """One peer's observed multiple."""

    entity_id: str = Field(min_length=1)
    name: str
    multiple: Decimal
    basis: str = Field(min_length=1, description="What the multiple is of, e.g. 'ebitda'")


@dataclass(frozen=True)
class MarketData:
    """Observable market inputs.

    Facts about a price at a moment, not judgements — but a stale price makes a
    valuation quietly wrong, so the observation date travels with them.
    """

    share_price: Money
    diluted_shares: Shares
    net_debt: Money | None = None

    @property
    def market_capitalisation(self) -> Money:
        return self.share_price.scaled_by(self.diluted_shares.count)

    @property
    def enterprise_value(self) -> Money | None:
        if self.net_debt is None:
            return None
        return self.market_capitalisation + self.net_debt


def _median(values: list[Decimal]) -> Decimal:
    """Median rather than mean: one peer trading at 200x should not set the bar."""
    if not values:
        raise ValidationError("cannot take a median of an empty peer set")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return ((ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2)).quantize(
        _PRECISION, rounding=ROUND_HALF_UP
    )


def price_to_earnings(market: MarketData, net_income: Money) -> Metric:
    """Market capitalisation over net income."""
    computation = f"atlas.comparables.price_to_earnings.{_VERSION}"
    if net_income.is_zero:
        return unavailable(
            "price_to_earnings",
            Unit.TIMES,
            reason="net income is zero, so the multiple is undefined",
            computation=computation,
        )
    if net_income.is_negative:
        # A negative P/E is not a cheap stock; the multiple simply does not
        # apply to a loss-making company, and printing one invites misreading.
        return unavailable(
            "price_to_earnings",
            Unit.TIMES,
            reason="net income is negative; a price-to-earnings multiple does not apply",
            computation=computation,
        )

    multiple = market.market_capitalisation.ratio_to(net_income)
    return derived(
        "price_to_earnings",
        multiple.quantize(_PRECISION, rounding=ROUND_HALF_UP),
        Unit.TIMES,
        computation=computation,
        inputs={
            "market_capitalisation": market.market_capitalisation,
            "net_income": net_income,
        },
    )


def enterprise_value_to_ebitda(market: MarketData, ebitda: Money) -> Metric:
    """Enterprise value over EBITDA."""
    computation = f"atlas.comparables.ev_to_ebitda.{_VERSION}"
    enterprise_value = market.enterprise_value
    if enterprise_value is None:
        return unavailable(
            "ev_to_ebitda",
            Unit.TIMES,
            reason="net debt is required to compute enterprise value",
            computation=computation,
        )
    if ebitda.is_zero or ebitda.is_negative:
        return unavailable(
            "ev_to_ebitda",
            Unit.TIMES,
            reason="EBITDA must be positive for the multiple to be meaningful",
            computation=computation,
        )

    multiple = enterprise_value.ratio_to(ebitda)
    return derived(
        "ev_to_ebitda",
        multiple.quantize(_PRECISION, rounding=ROUND_HALF_UP),
        Unit.TIMES,
        computation=computation,
        inputs={"enterprise_value": enterprise_value, "ebitda": ebitda},
    )


def price_to_sales(market: MarketData, revenue: Money) -> Metric:
    computation = f"atlas.comparables.price_to_sales.{_VERSION}"
    if revenue.is_zero or revenue.is_negative:
        return unavailable(
            "price_to_sales",
            Unit.TIMES,
            reason="revenue must be positive for the multiple to be meaningful",
            computation=computation,
        )
    multiple = market.market_capitalisation.ratio_to(revenue)
    return derived(
        "price_to_sales",
        multiple.quantize(_PRECISION, rounding=ROUND_HALF_UP),
        Unit.TIMES,
        computation=computation,
        inputs={"market_capitalisation": market.market_capitalisation, "revenue": revenue},
    )


class ComparableValuation(FrozenModel):
    """An implied value from the peer set's median multiple."""

    entity_id: str
    basis: str
    peer_count: int = Field(ge=1)
    median_multiple: Decimal
    peer_multiples: dict[str, str] = Field(default_factory=dict)
    implied_value: Money
    implied_value_per_share: Money | None = None

    @property
    def peer_set_is_thin(self) -> bool:
        """Whether the peer set is too small for the median to mean much."""
        return self.peer_count < 3


def value_from_peers(
    entity_id: str,
    *,
    peers: list[PeerMultiple],
    metric_value: Money,
    basis: str,
    diluted_shares: Shares | None = None,
    net_debt: Money | None = None,
) -> ComparableValuation:
    """Apply the peer set's median multiple to this company's metric.

    Args:
        metric_value: This company's figure for ``basis`` — its EBITDA when
            valuing on EV/EBITDA, its net income for P/E.
        net_debt: Subtracted when the multiple produces an enterprise value, so
            the result is comparable to an equity value.

    Raises:
        ValidationError: if the peer set is empty or mixes bases. Applying an
            EV/EBITDA multiple to net income produces a number with no meaning.
    """
    if not peers:
        raise ValidationError(
            "a comparable valuation needs at least one peer",
            details={"entity_id": entity_id, "basis": basis},
        )

    mismatched = {peer.basis for peer in peers} - {basis}
    if mismatched:
        raise ValidationError(
            "every peer multiple must be on the same basis as the valuation",
            details={"basis": basis, "mismatched": sorted(mismatched)},
        )
    if metric_value.is_zero or metric_value.is_negative:
        raise ValidationError(
            f"cannot value on {basis} when the company's own {basis} is not positive",
            details={"entity_id": entity_id, "value": str(metric_value)},
        )

    median = _median([peer.multiple for peer in peers])
    implied = metric_value.scaled_by(median)

    per_share = None
    equity_value = implied
    if net_debt is not None:
        equity_value = implied - net_debt
    if diluted_shares is not None:
        per_share = equity_value.divided_by(diluted_shares.count).rounded(Decimal("0.0001"))

    return ComparableValuation(
        entity_id=entity_id,
        basis=basis,
        peer_count=len(peers),
        median_multiple=median,
        peer_multiples={peer.name: str(peer.multiple) for peer in peers},
        implied_value=implied.rounded(),
        implied_value_per_share=per_share,
    )


def to_metric(valuation: ComparableValuation) -> Metric:
    """Express a comparable valuation as an attributed metric.

    An ESTIMATE, not a DERIVED value: the peer set is a choice, and a different
    set gives a different answer. Recording the peers as assumptions is what
    lets a reader disagree with the selection rather than only the result.
    """
    return estimated(
        f"comparable_value_{valuation.basis}",
        (valuation.implied_value_per_share or valuation.implied_value).amount,
        Unit.MONEY,
        assumptions={
            "basis": valuation.basis,
            "median_multiple": valuation.median_multiple,
            "peer_count": valuation.peer_count,
            "peers": ", ".join(sorted(valuation.peer_multiples)),
        },
        currency=valuation.implied_value.currency,
    )


__all__ = [
    "ComparableValuation",
    "MarketData",
    "PeerMultiple",
    "enterprise_value_to_ebitda",
    "price_to_earnings",
    "price_to_sales",
    "to_metric",
    "value_from_peers",
]
