"""fie_schemas — shared contracts crossing service boundaries."""

from fie_schemas.base import (
    FIEModel,
    FrozenModel,
    IdentifiedModel,
    IdentifiedTimestampedModel,
    TimestampedModel,
)
from fie_schemas.envelope import (
    ErrorDetail,
    Page,
    PageInfo,
    PageRequest,
    ResponseEnvelope,
)
from fie_schemas.health import ComponentHealth, HealthReport, HealthStatus
from fie_schemas.provenance import (
    AssertionKind,
    Attributed,
    Provenance,
    SourceLocator,
    SourceReference,
    SourceType,
)

__version__ = "0.1.0"

__all__ = [
    "AssertionKind",
    "Attributed",
    "ComponentHealth",
    "ErrorDetail",
    "FIEModel",
    "FrozenModel",
    "HealthReport",
    "HealthStatus",
    "IdentifiedModel",
    "IdentifiedTimestampedModel",
    "Page",
    "PageInfo",
    "PageRequest",
    "Provenance",
    "ResponseEnvelope",
    "SourceLocator",
    "SourceReference",
    "SourceType",
    "TimestampedModel",
    "__version__",
]
