"""Atlas configuration.

Everything resolves from ``ATLAS_``-prefixed environment variables. The shared
infrastructure settings (Postgres, Redis, AI providers) come from their own
packages, so this class holds only what is specific to filing analysis and
research.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from atlas.analysis.dcf import MINIMUM_SPREAD
from fie_common.config import Environment, FIEBaseSettings
from fie_common.errors import ConfigurationError


class AtlasSettings(FIEBaseSettings):
    """Atlas service configuration (``ATLAS_*``)."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    service_name: str = "atlas"
    #: Loopback by default. Binding every interface is a deployment decision,
    #: not a default: the container image opts in explicitly by setting
    #: ATLAS_API_HOST, so a developer running the service locally does not
    #: silently expose it to their network.
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8002, ge=1, le=65535)

    # -- extraction ----------------------------------------------------------
    #: Characters of a filing sent to the model. Statements sit in the financial
    #: section; sending a whole 10-K buries the tables among the risk factors
    #: and costs context for nothing.
    extraction_max_content_chars: int = Field(default=60_000, ge=1_000)
    extraction_max_tokens: int = Field(default=4096, ge=256)

    # -- research ------------------------------------------------------------
    report_max_tokens: int = Field(default=4096, ge=256)
    #: How many times an ungrounded draft is regenerated before the report is
    #: returned unpublishable. One retry, with the offending figures named, is
    #: corrective; more is rerolling the same dice and paying for it.
    report_max_regenerations: int = Field(default=1, ge=0, le=3)

    # -- valuation defaults --------------------------------------------------
    #: Starting points for a DCF when a caller supplies none. Defaults, not
    #: recommendations: every valuation records the assumptions it actually
    #: used, and these appear there like any other stated input.
    default_discount_rate: Decimal = Field(default=Decimal("0.10"), gt=0, lt=1)
    default_terminal_growth: Decimal = Field(default=Decimal("0.025"), ge=0, lt=1)
    default_forecast_years: int = Field(default=5, ge=1, le=20)

    # -- MarketMind ----------------------------------------------------------
    #: Company context — competitors, suppliers, executives — for the narrative.
    #: Atlas degrades to no context rather than failing when it is unreachable;
    #: see atlas.clients.marketmind.
    marketmind_base_url: str = "http://localhost:8001"
    marketmind_timeout_seconds: float = Field(default=10.0, gt=0)
    marketmind_enabled: bool = True
    #: Service token for calling MarketMind. Never defaulted to a real value;
    #: absent means Atlas calls unauthenticated, which MarketMind will reject.
    marketmind_token: str | None = None

    # -- events --------------------------------------------------------------
    events_consumer_group: str = "atlas"
    publish_events: bool = True

    def validate_for(self, environment: Environment) -> None:
        """Fail at startup rather than at the first valuation.

        Raises:
            ConfigurationError: on a discount rate that does not clear terminal
                growth, or on a localhost MarketMind URL in production.
        """
        spread = self.default_discount_rate - self.default_terminal_growth
        if spread < MINIMUM_SPREAD:
            raise ConfigurationError(
                "ATLAS_DEFAULT_DISCOUNT_RATE must exceed ATLAS_DEFAULT_TERMINAL_GROWTH "
                f"by at least {MINIMUM_SPREAD}; a terminal value computed from a "
                "narrower spread diverges toward infinity",
                details={
                    "discount_rate": str(self.default_discount_rate),
                    "terminal_growth": str(self.default_terminal_growth),
                    "spread": str(spread),
                },
            )

        if (
            environment.is_production
            and self.marketmind_enabled
            and self.marketmind_base_url.startswith(("http://localhost", "http://127.0.0.1"))
        ):
            raise ConfigurationError(
                "ATLAS_MARKETMIND_BASE_URL still points at localhost in production",
                details={"base_url": self.marketmind_base_url},
            )

        if environment.is_production and self.marketmind_enabled and not self.marketmind_token:
            raise ConfigurationError(
                "ATLAS_MARKETMIND_TOKEN is required in production; MarketMind rejects "
                "unauthenticated callers, so every context lookup would fail silently",
                details={"base_url": self.marketmind_base_url},
            )


__all__ = ["AtlasSettings"]
