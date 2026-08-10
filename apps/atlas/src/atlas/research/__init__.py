"""Research: narrative written from computed figures, and the checks on it."""

from atlas.research.grounding import (
    NumericGroundingReport,
    check_numeric_grounding,
    extract_numeric_claims,
)
from atlas.research.reports import (
    ReportRequest,
    ReportSection,
    ReportService,
    ResearchReport,
)

__all__ = [
    "NumericGroundingReport",
    "ReportRequest",
    "ReportSection",
    "ReportService",
    "ResearchReport",
    "check_numeric_grounding",
    "extract_numeric_claims",
]
