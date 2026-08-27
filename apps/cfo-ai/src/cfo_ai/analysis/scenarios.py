"""Scenario modelling.

A scenario is a baseline plan plus a stated set of overrides — "revenue down
20%", "freeze hiring in Engineering", "delay the office move by two quarters" —
recomputed deterministically. The arithmetic is trivial. What makes it useful is
that the result records exactly which drivers were changed and by how much, so
two scenarios can be compared on their assumptions rather than on their
conclusions.

Nothing here consults a model. A scenario whose numbers came from a language
model would be a guess with a spreadsheet's authority, which is the specific
failure this ecosystem is built to refuse.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

from pydantic import Field, model_validator

from cfo_ai.domain.plan import Plan, PlanKind, PlanLine
from fie_common.errors import ValidationError
from fie_finance.metrics import Metric, Unit, estimated
from fie_finance.money import Money
from fie_schemas.base import FrozenModel
from fie_schemas.provenance import Provenance, SourceReference

COMPUTATION = "cfo_ai.scenario.v1"

_MONEY_EXPONENT = Decimal("0.01")


class DriverScope(StrEnum):
    """What a driver applies to."""

    ACCOUNT = "account"
    COST_CENTRE = "cost_centre"
    #: Every line in the plan.
    ALL = "all"


class Driver(FrozenModel):
    """One stated change to the baseline.

    Either a multiplier or an absolute delta, never both — "cut marketing by
    20%" and "cut marketing by $50k" are different instructions, and a driver
    that silently carried both would apply them in an order nobody specified.
    """

    scope: DriverScope
    #: Account code or cost centre id. Ignored when scope is ALL.
    target: str | None = None
    #: ``0.8`` cuts by twenty percent; ``1.1`` grows by ten.
    multiplier: Decimal | None = None
    #: Added after any multiplier, in the plan's currency.
    delta: Decimal | None = None
    #: Why. Required — a scenario driver nobody can explain is a number nobody
    #: should plan against.
    rationale: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def _validate(self) -> Driver:
        if self.multiplier is None and self.delta is None:
            raise ValueError("a driver must state a multiplier or a delta")
        if self.multiplier is not None and self.delta is not None:
            raise ValueError(
                "a driver states a multiplier or a delta, not both; applying both "
                "requires an order nobody specified"
            )
        if self.multiplier is not None and self.multiplier < 0:
            raise ValueError(
                "a negative multiplier flips the sign of a line rather than scaling it"
            )
        if self.scope is not DriverScope.ALL and not self.target:
            raise ValueError(f"a {self.scope} driver must name its target")
        return self

    def applies_to(self, line: PlanLine) -> bool:
        if self.scope is DriverScope.ALL:
            return True
        if self.scope is DriverScope.ACCOUNT:
            return line.account_code == (self.target or "").strip().upper()
        return line.cost_centre_id == self.target

    def describe(self) -> str:
        target = self.target or "every line"
        if self.multiplier is not None:
            change = f"x{self.multiplier}"
        else:
            change = f"{'+' if (self.delta or 0) >= 0 else ''}{self.delta}"
        return f"{target}: {change} ({self.rationale})"


class Scenario(FrozenModel):
    """A named set of drivers applied to a baseline."""

    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    drivers: list[Driver] = Field(min_length=1)

    def assumptions(self) -> dict[str, str]:
        """The drivers, rendered for ESTIMATE provenance."""
        return {
            f"driver_{index}": driver.describe() for index, driver in enumerate(self.drivers, 1)
        }


class ScenarioResult(FrozenModel):
    """What a scenario does to a baseline."""

    scenario_name: str
    baseline_total: Money
    scenario_total: Money
    #: ``scenario - baseline``.
    change: Money
    #: The recomputed plan, so a caller can compare line by line rather than
    #: only in aggregate.
    plan: Plan
    #: Lines no driver touched. Surfaced because a scenario that quietly missed
    #: its target reads exactly like one that had no effect.
    untouched_line_count: int = Field(ge=0)
    assumptions: dict[str, str] = Field(default_factory=dict)

    @property
    def changed_line_count(self) -> int:
        return len(self.plan.lines) - self.untouched_line_count

    @property
    def had_no_effect(self) -> bool:
        """True when no line matched any driver — almost always a typo in a target."""
        return self.changed_line_count == 0

    def to_metric(self, sources: list[SourceReference] | None = None) -> Metric:
        return estimated(
            f"scenario.{self.scenario_name}.total",
            self.scenario_total.amount,
            Unit.MONEY,
            assumptions=self.assumptions,
            currency=self.scenario_total.currency,
            sources=sources,
            inputs={
                "baseline_total": str(self.baseline_total.amount),
                "lines_changed": str(self.changed_line_count),
            },
        )


def apply_scenario(baseline: Plan, scenario: Scenario) -> ScenarioResult:
    """Recompute a plan under a scenario.

    The baseline is never mutated; a new plan is returned, marked as a
    REFORECAST and carrying ESTIMATE provenance. A scenario stored as a budget
    would eventually be compared against actuals as though someone had approved
    it.

    Raises:
        ValidationError: if a driver names a target that appears nowhere in the
            baseline. Silently producing an unchanged plan would let a mistyped
            cost centre look like a scenario with no impact.
    """
    known_accounts = set(baseline.account_codes)
    known_centres = set(baseline.cost_centre_ids)
    for driver in scenario.drivers:
        if driver.scope is DriverScope.ACCOUNT:
            target = (driver.target or "").strip().upper()
            if target not in known_accounts:
                raise ValidationError(
                    "a scenario driver names an account that is not in the baseline",
                    details={"account_code": target, "scenario": scenario.name},
                )
        elif driver.scope is DriverScope.COST_CENTRE and driver.target not in known_centres:
            raise ValidationError(
                "a scenario driver names a cost centre that is not in the baseline",
                details={"cost_centre_id": driver.target, "scenario": scenario.name},
            )

    new_lines: list[PlanLine] = []
    untouched = 0

    for line in baseline.lines:
        applicable = [driver for driver in scenario.drivers if driver.applies_to(line)]
        if not applicable:
            untouched += 1
            new_lines.append(line)
            continue

        amount = line.amount.amount
        for driver in applicable:
            if driver.multiplier is not None:
                amount = amount * driver.multiplier
            if driver.delta is not None:
                amount = amount + driver.delta

        new_lines.append(
            line.model_copy(
                update={
                    "amount": Money(
                        amount=amount.quantize(_MONEY_EXPONENT, rounding=ROUND_HALF_UP),
                        currency=line.amount.currency,
                    )
                }
            )
        )

    projected = Plan(
        entity_id=baseline.entity_id,
        kind=PlanKind.REFORECAST,
        period=baseline.period,
        currency=baseline.currency,
        version=f"{baseline.version}+{scenario.name}",
        lines=new_lines,
        provenance=Provenance.estimate(scenario.assumptions(), *baseline.sources),
    )

    return ScenarioResult(
        scenario_name=scenario.name,
        baseline_total=baseline.total(),
        scenario_total=projected.total(),
        change=projected.total() - baseline.total(),
        plan=projected,
        untouched_line_count=untouched,
        assumptions=scenario.assumptions(),
    )


def compare_scenarios(results: list[ScenarioResult]) -> list[ScenarioResult]:
    """Order scenarios by total, cheapest first.

    Deliberately does not pick a winner. Which scenario is *best* depends on
    risk appetite, strategy, and what the board will tolerate — none of which
    are in this data, and a service that ranked them would be asserting a
    judgement it has no basis for.
    """
    return sorted(results, key=lambda result: result.scenario_total.amount)


__all__ = [
    "COMPUTATION",
    "Driver",
    "DriverScope",
    "Scenario",
    "ScenarioResult",
    "apply_scenario",
    "compare_scenarios",
]
