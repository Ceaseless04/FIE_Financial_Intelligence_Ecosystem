"""Reading financial statements out of a filing.

This is the one place in Atlas where a language model touches numbers, and the
design assumes it will sometimes get them wrong.

Two properties keep that from mattering:

1. **The model transcribes; it never calculates.** Figures come back as strings
   exactly as printed, along with the scale the statement is presented in
   ("in millions"). Atlas applies the scale itself. Asking a model to multiply
   by a million is asking it to do arithmetic, which is precisely what the
   architecture forbids.
2. **The result is checked against accounting identities.** Assets must equal
   liabilities plus equity; gross profit must equal revenue less cost of
   revenue; the cash flow statement must reconcile. These are true of every
   filing ever published, so a violation means the extraction is wrong — and
   the statement is rejected rather than stored.

The second point is what makes a model usable here at all. Transcription is
checkable, and a transcription that fails its check is discarded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from pydantic import Field
from pydantic import ValidationError as PydanticValidationError

from atlas.domain.money import Money, Shares, to_decimal
from atlas.domain.periods import FiscalPeriod
from atlas.domain.statements import (
    BalanceSheet,
    CashFlowStatement,
    FinancialStatements,
    IncomeStatement,
)
from atlas.filings.models import Filing
from fie_ai import AIRouter, CompletionRequest, Effort, Message, Role
from fie_ai.structured import parse_structured, structured_request
from fie_common.errors import AIProviderResponseError
from fie_observability.logging import get_logger
from fie_schemas.base import FIEModel
from fie_schemas.provenance import Provenance

logger = get_logger(__name__)

#: Converts a transcribed figure into a scaled monetary amount, or None when
#: the line item was absent or unparsable.
MoneyParser = Callable[[str | None], "Money | None"]


class ReportingScale(StrEnum):
    """The unit a statement is presented in.

    Applied by Atlas, never by the model. "In thousands" at the top of a table
    is the single most consequential piece of context in a filing, and a model
    that both reads it and applies it can be wrong by three orders of magnitude
    without anything looking unusual.
    """

    UNITS = "units"
    THOUSANDS = "thousands"
    MILLIONS = "millions"
    BILLIONS = "billions"

    @property
    def multiplier(self) -> Decimal:
        return {
            ReportingScale.UNITS: Decimal(1),
            ReportingScale.THOUSANDS: Decimal(1_000),
            ReportingScale.MILLIONS: Decimal(1_000_000),
            ReportingScale.BILLIONS: Decimal(1_000_000_000),
        }[self]


EXTRACTION_SYSTEM_PROMPT = """\
You transcribe financial statements from filings into a structured form.

You are a transcriber, not a calculator. Copy each figure exactly as printed.

Rules:
- Report every figure as a string, exactly as it appears, without thousands \
separators and without currency symbols. "1,234.5" becomes "1234.5".
- Never compute, sum, convert, or scale a figure. If the statement says the \
table is in millions, report the printed number and set the scale field; do \
not multiply it yourself.
- Report a figure shown in parentheses as negative: "(1234)" becomes "-1234".
- If a line item is not present, leave it null. Do not infer it, and do not \
derive it from other lines.
- Report the currency and the reporting scale exactly as the statement states \
them.
- Do not round, and do not tidy up a number that looks wrong. A figure that \
does not reconcile is information; a corrected one is a fabrication."""


class ExtractedIncomeStatement(FIEModel):
    """Income statement figures as printed. Every value is a string."""

    revenue: str
    cost_of_revenue: str | None = None
    gross_profit: str | None = None
    operating_expenses: str | None = None
    operating_income: str | None = None
    interest_expense: str | None = None
    pretax_income: str | None = None
    income_tax_expense: str | None = None
    net_income: str
    diluted_shares: str | None = None


class ExtractedBalanceSheet(FIEModel):
    total_assets: str
    current_assets: str | None = None
    cash_and_equivalents: str | None = None
    inventory: str | None = None
    total_liabilities: str
    current_liabilities: str | None = None
    total_debt: str | None = None
    shareholders_equity: str


class ExtractedCashFlow(FIEModel):
    operating_cash_flow: str
    capital_expenditures: str | None = None
    investing_cash_flow: str | None = None
    financing_cash_flow: str | None = None
    net_change_in_cash: str | None = None


class ExtractedStatements(FIEModel):
    """The model's structured response for one filing."""

    currency: str = Field(default="USD", min_length=3, max_length=3)
    scale: ReportingScale = ReportingScale.UNITS
    income_statement: ExtractedIncomeStatement | None = None
    balance_sheet: ExtractedBalanceSheet | None = None
    cash_flow_statement: ExtractedCashFlow | None = None


@dataclass
class ExtractionOutcome:
    """What came out of a filing, and what was refused.

    Rejections are returned rather than logged and dropped. A filing whose
    balance sheet failed its identity check is a fact worth surfacing: it means
    either the extraction or the source is unusable, and a caller that silently
    receives fewer statements cannot tell which.
    """

    filing_id: str
    statements: FinancialStatements | None = None
    rejected: list[str] = field(default_factory=list)
    #: Set when the figures were transcribed but failed validation, as opposed
    #: to never having been found at all.
    failed_validation: bool = False

    @property
    def succeeded(self) -> bool:
        return self.statements is not None

    @property
    def rejection_count(self) -> int:
        return len(self.rejected)


class StatementExtractor:
    """Transcribes statements from a filing and validates them arithmetically."""

    def __init__(
        self,
        router: AIRouter,
        *,
        max_tokens: int = 4096,
        effort: Effort | None = Effort.MEDIUM,
        max_content_chars: int = 60_000,
    ) -> None:
        self._router = router
        self._max_tokens = max_tokens
        self._effort = effort
        self._max_content_chars = max_content_chars

    async def extract(self, filing: Filing) -> ExtractionOutcome:
        """Read the statements out of a filing.

        Raises:
            AIProviderResponseError: if the provider refuses or returns output
                that does not satisfy the schema. A caller must be able to tell
                "the model would not answer" from "the filing has no balance
                sheet".
        """
        outcome = ExtractionOutcome(filing_id=filing.id)

        try:
            extracted = await self._transcribe(filing)
        except AIProviderResponseError as error:
            logger.warning("statement_extraction_failed", filing_id=filing.id, error=str(error))
            outcome.rejected.append(f"transcription failed: {error}")
            return outcome

        return self._validate(filing, extracted, outcome)

    async def _transcribe(self, filing: Filing) -> ExtractedStatements:
        request = structured_request(
            CompletionRequest(
                messages=[Message(role=Role.USER, content=self._build_prompt(filing))],
                system=EXTRACTION_SYSTEM_PROMPT,
                max_tokens=self._max_tokens,
                effort=self._effort,
            ),
            ExtractedStatements,
        )
        response = await self._router.complete(request)
        return parse_structured(response, ExtractedStatements)

    def _build_prompt(self, filing: Filing) -> str:
        # Statements sit in the financial section; sending a whole 10-K wastes
        # context and buries the tables among the risk factors.
        content = filing.content[: self._max_content_chars]
        truncated = len(filing.content) > self._max_content_chars
        parts = [
            f"Filing type: {filing.type}",
            f"Period: {filing.period.label}",
            f"Period end: {filing.period.end_date.isoformat()}",
            "",
            "Transcribe the financial statements from the filing below.",
            "",
            content,
        ]
        if truncated:
            parts.append("")
            parts.append("[content truncated]")
        return "\n".join(parts)

    # -- validation ----------------------------------------------------------

    def _validate(
        self, filing: Filing, extracted: ExtractedStatements, outcome: ExtractionOutcome
    ) -> ExtractionOutcome:
        """Convert transcribed strings into validated domain statements.

        Constructing the domain models *is* the validation: they refuse to exist
        unless the accounting identities hold.
        """
        multiplier = extracted.scale.multiplier
        currency = extracted.currency.upper()
        source = filing.to_source_reference()
        provenance = Provenance.fact(source)

        def money(value: str | None) -> Money | None:
            if value is None or not value.strip():
                return None
            try:
                return Money(amount=to_decimal(value.strip()) * multiplier, currency=currency)
            except (ValueError, TypeError) as error:
                outcome.rejected.append(f"unparsable figure {value!r}: {error}")
                return None

        income = self._build_income(filing, extracted, money, provenance, currency, outcome)
        balance = self._build_balance(filing, extracted, money, provenance, currency, outcome)
        cash_flow = self._build_cash_flow(filing, extracted, money, provenance, currency, outcome)

        if income is None and balance is None and cash_flow is None:
            if filing.type.carries_full_statements and not outcome.rejected:
                outcome.rejected.append(
                    "no financial statements were found in a filing type that should contain them"
                )
            return outcome

        try:
            outcome.statements = FinancialStatements(
                entity_id=filing.entity_id,
                period=filing.period,
                currency=currency,
                income_statement=income,
                balance_sheet=balance,
                cash_flow_statement=cash_flow,
            )
        except PydanticValidationError as error:
            outcome.rejected.append(f"statement set rejected: {_first_message(error)}")
            outcome.failed_validation = True

        logger.info(
            "statements_extracted",
            filing_id=filing.id,
            entity_id=filing.entity_id,
            succeeded=outcome.succeeded,
            rejected=outcome.rejection_count,
            scale=str(extracted.scale),
        )
        return outcome

    @staticmethod
    def _build_income(
        filing: Filing,
        extracted: ExtractedStatements,
        money: MoneyParser,
        provenance: Provenance,
        currency: str,
        outcome: ExtractionOutcome,
    ) -> IncomeStatement | None:
        raw = extracted.income_statement
        if raw is None:
            return None

        revenue = money(raw.revenue)
        net_income = money(raw.net_income)
        if revenue is None or net_income is None:
            outcome.rejected.append("income statement requires both revenue and net income")
            return None

        shares = None
        if raw.diluted_shares:
            try:
                shares = Shares.of(to_decimal(raw.diluted_shares.strip()))
            except (ValueError, TypeError, PydanticValidationError):
                # A malformed share count costs per-share metrics, not the
                # whole statement.
                outcome.rejected.append(f"unusable diluted share count {raw.diluted_shares!r}")

        try:
            return IncomeStatement(
                period=filing.period,
                currency=currency,
                provenance=provenance,
                revenue=revenue,
                cost_of_revenue=money(raw.cost_of_revenue),
                gross_profit=money(raw.gross_profit),
                operating_expenses=money(raw.operating_expenses),
                operating_income=money(raw.operating_income),
                interest_expense=money(raw.interest_expense),
                pretax_income=money(raw.pretax_income),
                income_tax_expense=money(raw.income_tax_expense),
                net_income=net_income,
                diluted_shares=shares,
            )
        except PydanticValidationError as error:
            # The arithmetic check caught a transcription error.
            outcome.rejected.append(f"income statement rejected: {_first_message(error)}")
            outcome.failed_validation = True
            return None

    @staticmethod
    def _build_balance(
        filing: Filing,
        extracted: ExtractedStatements,
        money: MoneyParser,
        provenance: Provenance,
        currency: str,
        outcome: ExtractionOutcome,
    ) -> BalanceSheet | None:
        raw = extracted.balance_sheet
        if raw is None:
            return None

        total_assets = money(raw.total_assets)
        total_liabilities = money(raw.total_liabilities)
        equity = money(raw.shareholders_equity)
        if total_assets is None or total_liabilities is None or equity is None:
            outcome.rejected.append(
                "balance sheet requires total assets, total liabilities, and equity"
            )
            return None

        try:
            return BalanceSheet(
                # A balance sheet is a position at the period end, not a span.
                period=FiscalPeriod.instant(filing.period.fiscal_year, filing.period.end_date),
                currency=currency,
                provenance=provenance,
                total_assets=total_assets,
                current_assets=money(raw.current_assets),
                cash_and_equivalents=money(raw.cash_and_equivalents),
                inventory=money(raw.inventory),
                total_liabilities=total_liabilities,
                current_liabilities=money(raw.current_liabilities),
                total_debt=money(raw.total_debt),
                shareholders_equity=equity,
            )
        except PydanticValidationError as error:
            outcome.rejected.append(f"balance sheet rejected: {_first_message(error)}")
            outcome.failed_validation = True
            return None

    @staticmethod
    def _build_cash_flow(
        filing: Filing,
        extracted: ExtractedStatements,
        money: MoneyParser,
        provenance: Provenance,
        currency: str,
        outcome: ExtractionOutcome,
    ) -> CashFlowStatement | None:
        raw = extracted.cash_flow_statement
        if raw is None:
            return None

        operating = money(raw.operating_cash_flow)
        if operating is None:
            outcome.rejected.append("cash flow statement requires operating cash flow")
            return None

        try:
            return CashFlowStatement(
                period=filing.period,
                currency=currency,
                provenance=provenance,
                operating_cash_flow=operating,
                capital_expenditures=money(raw.capital_expenditures),
                investing_cash_flow=money(raw.investing_cash_flow),
                financing_cash_flow=money(raw.financing_cash_flow),
                net_change_in_cash=money(raw.net_change_in_cash),
            )
        except PydanticValidationError as error:
            outcome.rejected.append(f"cash flow statement rejected: {_first_message(error)}")
            outcome.failed_validation = True
            return None


def _first_message(error: PydanticValidationError) -> str:
    """The first validation message, without Pydantic's surrounding noise."""
    errors = error.errors()
    if not errors:
        return str(error)
    message = str(errors[0].get("msg", ""))
    return message.removeprefix("Value error, ")


def balance_sheet_residual(balance: BalanceSheet) -> Money:
    """How far the accounting identity is from closing.

    Exposed so a caller can report the size of a discrepancy rather than only
    that one exists — a residual of a few dollars on a billion-dollar balance
    sheet is rounding; a residual of ten percent is a misread column.
    """
    return (balance.total_liabilities + balance.shareholders_equity) - balance.total_assets


__all__ = [
    "EXTRACTION_SYSTEM_PROMPT",
    "ExtractedBalanceSheet",
    "ExtractedCashFlow",
    "ExtractedIncomeStatement",
    "ExtractedStatements",
    "ExtractionOutcome",
    "ReportingScale",
    "StatementExtractor",
    "balance_sheet_residual",
]
