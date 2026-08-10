"""Deterministic entity resolution.

Identity decisions are never made by a language model — they must be
reproducible, auditable, and stable across re-ingestion.
"""

from marketmind.resolution.normalization import (
    blocking_key,
    name_tokens,
    normalize_company_name,
    normalize_generic,
    normalize_person_name,
    strip_accents,
)
from marketmind.resolution.resolver import (
    AUTHORITATIVE_IDENTIFIERS,
    EntityIndex,
    EntityResolver,
    MatchDecision,
    MatchReason,
    ResolutionConfig,
    canonicalize,
)

__all__ = [
    "AUTHORITATIVE_IDENTIFIERS",
    "EntityIndex",
    "EntityResolver",
    "MatchDecision",
    "MatchReason",
    "ResolutionConfig",
    "blocking_key",
    "canonicalize",
    "name_tokens",
    "normalize_company_name",
    "normalize_generic",
    "normalize_person_name",
    "strip_accents",
]
