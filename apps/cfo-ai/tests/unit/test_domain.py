"""Plan integrity and the shape of the organisation.

Both sets of invariants exist for the same reason: they fail silently. A
duplicated plan line still totals; a cost centre hierarchy with a dangling
parent still renders. Nobody notices until a number is wrong and nobody can say
why.
"""

from __future__ import annotations

import pytest
from cfo_fixtures import annual, fact, quarter
from pydantic import ValidationError as PydanticValidationError

from cfo_ai.domain.accounts import Account, AccountType
from cfo_ai.domain.organisation import CostCentre, CostCentreTree
from cfo_ai.domain.plan import Plan, PlanKind, line
from fie_common.errors import ValidationError
from fie_finance.money import Money

pytestmark = pytest.mark.unit


class TestAccounts:
    def test_a_code_must_look_like_a_ledger_code(self) -> None:
        with pytest.raises(PydanticValidationError, match="ledger account code"):
            Account(code="marketing", name="Marketing", type=AccountType.OPERATING_EXPENSE)

    def test_codes_are_normalised_so_lookups_cannot_miss(self) -> None:
        assert Account(code="6100a", name="X", type=AccountType.OPERATING_EXPENSE).code == "6100A"

    def test_aliases_are_deduplicated_and_lowercased(self) -> None:
        account = Account(
            code="4000",
            name="Revenue",
            type=AccountType.REVENUE,
            aliases=["Sales", "sales", " Top Line "],
        )
        assert account.aliases == ["sales", "top line"]

    def test_search_terms_include_the_name(self) -> None:
        account = Account(code="4000", name="Revenue", type=AccountType.REVENUE, aliases=["sales"])
        assert set(account.search_terms) == {"revenue", "sales"}

    @pytest.mark.parametrize(
        ("account_type", "inflow"),
        [
            (AccountType.REVENUE, True),
            (AccountType.MARGIN, True),
            (AccountType.CASH, True),
            (AccountType.OPERATING_EXPENSE, False),
            (AccountType.COST_OF_REVENUE, False),
            (AccountType.CAPITAL_EXPENDITURE, False),
        ],
    )
    def test_which_way_is_up(self, account_type: AccountType, inflow: bool) -> None:
        assert account_type.is_inflow is inflow

    def test_capex_and_cash_stay_out_of_the_operating_result(self) -> None:
        """Spending cash on a building is not an expense; folding it into an
        operating variance overstates the miss."""
        assert not AccountType.CAPITAL_EXPENDITURE.affects_profit
        assert not AccountType.CASH.affects_profit
        assert AccountType.OPERATING_EXPENSE.affects_profit


class TestCostCentreTree:
    def test_a_dangling_parent_is_refused(self) -> None:
        """Otherwise that centre's spend vanishes from every roll-up."""
        with pytest.raises(PydanticValidationError, match="does not exist"):
            CostCentreTree(
                centres=[
                    CostCentre(id="cc-co", name="Company"),
                    CostCentre(id="cc-eng", name="Engineering", parent_id="cc-missing"),
                ]
            )

    def test_a_cycle_is_refused(self) -> None:
        """A cycle makes every roll-up non-terminating."""
        with pytest.raises(PydanticValidationError, match="cycle"):
            CostCentreTree(
                centres=[
                    CostCentre(id="a", name="A", parent_id="b"),
                    CostCentre(id="b", name="B", parent_id="a"),
                ]
            )

    def test_a_centre_cannot_parent_itself(self) -> None:
        with pytest.raises(PydanticValidationError, match="its own parent"):
            CostCentre(id="a", name="A", parent_id="a")

    def test_duplicate_ids_are_refused(self) -> None:
        with pytest.raises(PydanticValidationError, match="duplicate"):
            CostCentreTree(
                centres=[CostCentre(id="a", name="A"), CostCentre(id="a", name="Also A")]
            )

    def test_a_rootless_hierarchy_is_caught_as_the_cycle_it_is(self) -> None:
        """There is no separate "needs a root" rule, because there cannot be one:
        if every parent resolves and no path repeats, walking upward always ends
        at a parentless centre. A rootless hierarchy is a cyclic one."""
        with pytest.raises(PydanticValidationError, match="cycle"):
            CostCentreTree(
                centres=[
                    CostCentre(id="a", name="A", parent_id="b"),
                    CostCentre(id="b", name="B", parent_id="c"),
                    CostCentre(id="c", name="C", parent_id="a"),
                ]
            )

    def test_ancestors_run_from_nearest_to_root(self) -> None:
        from cfo_fixtures import tree

        assert tree().ancestors_of("cc-field") == ["cc-sales", "cc-co"]

    def test_descendants_reach_every_depth(self) -> None:
        from cfo_fixtures import tree

        assert set(tree().descendants_of("cc-co")) == {"cc-eng", "cc-sales", "cc-field"}

    def test_leaves_are_the_centres_that_own_spend(self) -> None:
        from cfo_fixtures import tree

        assert {centre.id for centre in tree().leaves()} == {"cc-eng", "cc-field"}

    def test_an_unknown_centre_raises_rather_than_returning_nothing(self) -> None:
        from cfo_fixtures import tree

        with pytest.raises(ValidationError, match="unknown cost centre"):
            tree().ancestors_of("cc-nope")


class TestPlanIntegrity:
    def test_two_rows_for_one_key_are_refused(self) -> None:
        """Not extra detail — a double count that inflates every total silently."""
        period = annual()
        with pytest.raises(PydanticValidationError, match="duplicate plan line"):
            Plan(
                entity_id="ent-1",
                kind=PlanKind.BUDGET,
                period=period,
                lines=[
                    line("6100", "cc-sales", period, "100"),
                    line("6100", "cc-sales", period, "200"),
                ],
                provenance=fact(),
            )

    def test_the_same_account_in_two_centres_is_not_a_duplicate(self) -> None:
        period = annual()
        plan = Plan(
            entity_id="ent-1",
            kind=PlanKind.BUDGET,
            period=period,
            lines=[
                line("6100", "cc-sales", period, "100"),
                line("6100", "cc-eng", period, "200"),
            ],
            provenance=fact(),
        )
        assert plan.total() == Money.of("300")

    def test_a_line_outside_the_plan_period_is_refused(self) -> None:
        """A December actual in a November plan moves a miss into the wrong month."""
        with pytest.raises(PydanticValidationError, match="outside the plan period"):
            Plan(
                entity_id="ent-1",
                kind=PlanKind.BUDGET,
                period=quarter(2026, 1),
                lines=[line("6100", "cc-sales", quarter(2026, 3), "100")],
                provenance=fact(),
            )

    def test_a_quarterly_line_belongs_inside_an_annual_plan(self) -> None:
        """A yearly budget legitimately holds quarterly detail."""
        plan = Plan(
            entity_id="ent-1",
            kind=PlanKind.BUDGET,
            period=annual(2026),
            lines=[
                line("6100", "cc-sales", quarter(2026, 1), "100"),
                line("6100", "cc-sales", quarter(2026, 2), "100"),
            ],
            provenance=fact(),
        )
        assert plan.total() == Money.of("200")

    def test_a_line_in_another_currency_is_refused(self) -> None:
        period = annual()
        with pytest.raises(PydanticValidationError, match="convert explicitly"):
            Plan(
                entity_id="ent-1",
                kind=PlanKind.BUDGET,
                period=period,
                currency="USD",
                lines=[line("6100", "cc-sales", period, "100", currency="EUR")],
                provenance=fact(),
            )

    def test_a_plan_requires_provenance(self) -> None:
        """A budget nobody approved is a figure with no owner."""
        period = annual()
        with pytest.raises(PydanticValidationError):
            Plan(  # type: ignore[call-arg]
                entity_id="ent-1",
                kind=PlanKind.BUDGET,
                period=period,
                lines=[line("6100", "cc-sales", period, "100")],
            )

    def test_an_empty_plan_is_refused(self) -> None:
        with pytest.raises(PydanticValidationError):
            Plan(
                entity_id="ent-1",
                kind=PlanKind.BUDGET,
                period=annual(),
                lines=[],
                provenance=fact(),
            )

    def test_a_float_amount_never_enters(self) -> None:
        with pytest.raises(TypeError, match="float is not accepted"):
            line("6100", "cc-sales", annual(), 100.5)  # type: ignore[arg-type]

    def test_totals_by_account_and_centre(self, fy26_budget) -> None:
        assert fy26_budget.total_for_account("4000") == Money.of("4000000")
        assert fy26_budget.total_for_centre("cc-eng") == Money.of("3000000")

    def test_account_codes_are_normalised_on_lookup(self, fy26_budget) -> None:
        assert fy26_budget.lines_for_account("4000") == fy26_budget.lines_for_account(" 4000 ")
