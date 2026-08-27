"""Research: narrative written from computed figures, and the checks on it."""

from atlas.research.reports import (
    ReportRequest,
    ReportSection,
    ReportService,
    ResearchReport,
)
from fie_finance.grounding import (
    NumericGroundingReport,
    check_numeric_grounding,
    extract_numeric_claims,
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
