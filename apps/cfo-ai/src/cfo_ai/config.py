"""CFO.ai configuration.

Everything resolves from ``CFO_``-prefixed environment variables. Shared
infrastructure settings (Postgres, Redis, AI providers) come from their own
packages, so this class holds only what is specific to planning and analysis.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from fie_common.config import Environment, FIEBaseSettings
from fie_common.errors import ConfigurationError


class CFOSettings(FIEBaseSettings):
    """CFO.ai service configuration (``CFO_*``)."""

    model_config = SettingsConfigDict(
        env_prefix="CFO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    service_name: str = "cfo-ai"
    #: Loopback by default. Binding every interface is a deployment decision,
    #: not a default: the container image opts in explicitly by setting
    #: CFO_API_HOST, so a developer running the service locally does not
    #: silently expose a company's budget to their network.
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8003, ge=1, le=65535)

    # -- materiality ---------------------------------------------------------
    #: A line must clear both thresholds before it is worth a reader's
    #: attention. Percentage alone would rank 100% over on a $12 line beside a
    #: $2m overspend; amount alone would bury a small line that doubled.
    material_percent: Decimal = Field(default=Decimal("5"), gt=0, le=100)
    material_amount: Decimal = Field(default=Decimal("10000"), ge=0)

    # -- planning ------------------------------------------------------------
    #: Lines accepted in one plan upload. A general ledger export can be
    #: enormous, and an unbounded request is a memory profile nobody chose.
    max_plan_lines: int = Field(default=5000, ge=1, le=200_000)
    #: Periods a forecast may project. Beyond a couple of years every method
    #: here is extrapolating well past what its history supports.
    max_forecast_periods: int = Field(default=24, ge=1, le=120)

    # -- commentary ----------------------------------------------------------
    commentary_max_tokens: int = Field(default=3072, ge=256)
    #: How many times an ungrounded draft is regenerated before the commentary
    #: is returned unpublishable. One retry, naming the specific problems, is
    #: corrective; more is rerolling the same dice and paying for it.
    commentary_max_regenerations: int = Field(default=1, ge=0, le=3)

    # -- events --------------------------------------------------------------
    events_consumer_group: str = "cfo-ai"
    publish_events: bool = True

    def validate_for(self, environment: Environment) -> None:
        """Fail at startup rather than at the first variance report.

        Raises:
            ConfigurationError: on a materiality threshold that would make every
                line material, which is the same as having no threshold at all.
        """
        if self.material_percent <= 0:
            raise ConfigurationError(
                "CFO_MATERIAL_PERCENT must be positive; a zero threshold marks "
                "every line material and a report that flags everything flags nothing",
                details={"material_percent": str(self.material_percent)},
            )

        if environment.is_production and self.material_amount == 0:
            raise ConfigurationError(
                "CFO_MATERIAL_AMOUNT is zero in production, so a line that moved "
                "by a few pounds ranks alongside a six-figure overspend",
                details={"material_amount": str(self.material_amount)},
            )


__all__ = ["CFOSettings"]
