"""Entity resolution — deciding whether two records describe the same thing.

Entirely deterministic. A language model is not used to decide identity, for
three reasons: the decision must be reproducible across re-ingestion, it must
be auditable when someone asks why two companies were merged, and a wrong merge
silently corrupts every traversal that touches the node.

The decision ladder, strongest evidence first:

1. **Shared authoritative identifier** (CIK, LEI, ticker) — decisive.
2. **Conflicting authoritative identifier** — decisive *against* a match, even
   if the names are identical. Two companies both called "Acme Corp" with
   different CIKs are two companies.
3. **Identical canonical name** — strong.
4. **High fuzzy similarity** — probable, above a configured threshold.
5. Otherwise — a new entity.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from rapidfuzz import fuzz

from fie_schemas.base import FIEModel
from marketmind.domain.entities import Entity, EntityType, IdentifierType
from marketmind.resolution.normalization import (
    blocking_key,
    name_tokens,
    normalize_company_name,
    normalize_generic,
    normalize_person_name,
)

#: Identifiers strong enough to decide identity on their own.
AUTHORITATIVE_IDENTIFIERS: frozenset[IdentifierType] = frozenset(
    {IdentifierType.CIK, IdentifierType.LEI, IdentifierType.ISIN, IdentifierType.CUSIP}
)


class MatchReason(StrEnum):
    IDENTIFIER_MATCH = "identifier_match"
    EXACT_NAME_MATCH = "exact_name_match"
    ALIAS_MATCH = "alias_match"
    FUZZY_NAME_MATCH = "fuzzy_name_match"
    NO_MATCH = "no_match"
    IDENTIFIER_CONFLICT = "identifier_conflict"


class MatchDecision(FIEModel):
    """Why the resolver decided what it decided.

    Persisted alongside merges so a surprising merge can be explained without
    re-running the pipeline.
    """

    matched: bool
    reason: MatchReason
    confidence: float
    matched_entity_id: str | None = None
    #: Human-readable justification, e.g. the identifier or score that decided it.
    evidence: str = ""

    @property
    def is_automatic(self) -> bool:
        """Whether the decision is confident enough to merge without review."""
        return self.matched and self.confidence >= 0.9


@dataclass(frozen=True)
class ResolutionConfig:
    """Thresholds governing fuzzy matching.

    Deliberately conservative: a missed merge leaves two nodes that a later
    pass can still combine, while a wrong merge fuses two real companies into
    one and is very hard to undo once relationships hang off it.
    """

    #: Similarity at or above which two names are treated as the same entity.
    fuzzy_threshold: float = 92.0
    #: Below this, candidates are not even considered.
    minimum_candidate_score: float = 80.0
    #: Require this share of tokens in common before trusting a fuzzy score.
    #: Guards against short names scoring highly by accident ("Meta"/"Metro").
    minimum_token_overlap: float = 0.5


def canonicalize(name: str, entity_type: EntityType) -> str:
    """Normalize a name according to its entity type."""
    if entity_type is EntityType.COMPANY:
        return normalize_company_name(name)
    if entity_type is EntityType.EXECUTIVE:
        return normalize_person_name(name)
    return normalize_generic(name)


@dataclass
class EntityIndex:
    """In-memory index of known entities, blocked for cheap candidate lookup."""

    config: ResolutionConfig = field(default_factory=ResolutionConfig)
    _by_id: dict[str, Entity] = field(default_factory=dict)
    _by_identifier: dict[str, str] = field(default_factory=dict)
    _by_canonical: dict[tuple[EntityType, str], str] = field(default_factory=dict)
    _blocks: dict[tuple[EntityType, str], set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )

    def add(self, entity: Entity) -> Entity:
        """Index an entity, assigning its canonical name if absent."""
        canonical = entity.canonical_name or canonicalize(entity.name, entity.type)
        indexed = (
            entity
            if entity.canonical_name == canonical
            else entity.model_copy(update={"canonical_name": canonical})
        )

        self._by_id[indexed.id] = indexed
        for identifier in indexed.identifiers:
            self._by_identifier[identifier.key] = indexed.id
        self._by_canonical.setdefault((indexed.type, canonical), indexed.id)
        self._blocks[(indexed.type, blocking_key(canonical))].add(indexed.id)

        for alias in indexed.aliases:
            alias_canonical = canonicalize(alias, indexed.type)
            if alias_canonical:
                self._by_canonical.setdefault((indexed.type, alias_canonical), indexed.id)
                self._blocks[(indexed.type, blocking_key(alias_canonical))].add(indexed.id)

        return indexed

    def add_all(self, entities: Iterable[Entity]) -> list[Entity]:
        return [self.add(entity) for entity in entities]

    def get(self, entity_id: str) -> Entity | None:
        return self._by_id.get(entity_id)

    def replace(self, entity: Entity) -> None:
        """Update an indexed entity in place after a merge."""
        self._by_id[entity.id] = entity
        for identifier in entity.identifiers:
            self._by_identifier[identifier.key] = entity.id

    @property
    def entities(self) -> list[Entity]:
        return list(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def candidates(self, entity: Entity, canonical: str) -> list[Entity]:
        """Plausible matches for an entity, via identifier and name blocking."""
        candidate_ids: set[str] = set()

        for identifier in entity.identifiers:
            existing_id = self._by_identifier.get(identifier.key)
            if existing_id:
                candidate_ids.add(existing_id)

        candidate_ids |= self._blocks.get((entity.type, blocking_key(canonical)), set())

        exact_id = self._by_canonical.get((entity.type, canonical))
        if exact_id:
            candidate_ids.add(exact_id)

        return [self._by_id[candidate_id] for candidate_id in candidate_ids]


class EntityResolver:
    """Decides whether an incoming entity is already in the graph."""

    def __init__(self, config: ResolutionConfig | None = None) -> None:
        self.config = config or ResolutionConfig()

    def resolve(self, entity: Entity, index: EntityIndex) -> MatchDecision:
        """Match ``entity`` against the index, returning an auditable decision."""
        canonical = entity.canonical_name or canonicalize(entity.name, entity.type)
        candidates = index.candidates(entity, canonical)

        if not candidates:
            return MatchDecision(
                matched=False, reason=MatchReason.NO_MATCH, confidence=0.0, evidence="no candidates"
            )

        best_fuzzy: tuple[float, Entity] | None = None

        for candidate in candidates:
            shared, conflicting = self._compare_identifiers(entity, candidate)

            # A conflict is decisive against a match even when names agree — two
            # distinct legal entities can share a trading name.
            if conflicting:
                if shared:
                    # Genuinely ambiguous: some identifiers agree, others do not.
                    # Refuse to decide rather than guess.
                    return MatchDecision(
                        matched=False,
                        reason=MatchReason.IDENTIFIER_CONFLICT,
                        confidence=0.0,
                        matched_entity_id=candidate.id,
                        evidence=(f"shared {sorted(shared)} but conflicting {sorted(conflicting)}"),
                    )
                continue

            if shared:
                return MatchDecision(
                    matched=True,
                    reason=MatchReason.IDENTIFIER_MATCH,
                    confidence=1.0,
                    matched_entity_id=candidate.id,
                    evidence=f"shared identifier(s): {sorted(shared)}",
                )

            candidate_canonical = candidate.canonical_name or canonicalize(
                candidate.name, candidate.type
            )
            if canonical and canonical == candidate_canonical:
                return MatchDecision(
                    matched=True,
                    reason=MatchReason.EXACT_NAME_MATCH,
                    confidence=0.95,
                    matched_entity_id=candidate.id,
                    evidence=f"canonical name {canonical!r}",
                )

            if self._alias_matches(canonical, candidate):
                return MatchDecision(
                    matched=True,
                    reason=MatchReason.ALIAS_MATCH,
                    confidence=0.93,
                    matched_entity_id=candidate.id,
                    evidence=f"alias matches {canonical!r}",
                )

            score = self._similarity(canonical, candidate_canonical)
            if score >= self.config.minimum_candidate_score and (
                best_fuzzy is None or score > best_fuzzy[0]
            ):
                best_fuzzy = (score, candidate)

        if best_fuzzy is not None:
            score, candidate = best_fuzzy
            candidate_canonical = candidate.canonical_name or canonicalize(
                candidate.name, candidate.type
            )
            overlap = self._token_overlap(canonical, candidate_canonical)

            confident = score >= self.config.fuzzy_threshold
            shares_tokens = overlap >= self.config.minimum_token_overlap
            if confident and shares_tokens:
                return MatchDecision(
                    matched=True,
                    reason=MatchReason.FUZZY_NAME_MATCH,
                    confidence=round(min(score / 100.0, 0.94), 4),
                    matched_entity_id=candidate.id,
                    evidence=f"similarity {score:.1f}, token overlap {overlap:.2f}",
                )
            return MatchDecision(
                matched=False,
                reason=MatchReason.NO_MATCH,
                confidence=round(score / 100.0, 4),
                matched_entity_id=candidate.id,
                evidence=(
                    f"best similarity {score:.1f} below threshold "
                    f"{self.config.fuzzy_threshold} or token overlap {overlap:.2f} too low"
                ),
            )

        return MatchDecision(
            matched=False,
            reason=MatchReason.NO_MATCH,
            confidence=0.0,
            evidence=f"{len(candidates)} candidate(s), none similar enough",
        )

    def resolve_batch(
        self, entities: Sequence[Entity], index: EntityIndex
    ) -> tuple[list[Entity], list[MatchDecision]]:
        """Resolve a batch, merging matches and indexing new entities.

        Processing is sequential and order-dependent by design: entities within
        one batch can resolve against each other, so a document mentioning
        "Apple Inc." and later "Apple" produces one node.
        """
        resolved: list[Entity] = []
        decisions: list[MatchDecision] = []

        for entity in entities:
            decision = self.resolve(entity, index)
            decisions.append(decision)

            if decision.matched and decision.matched_entity_id:
                existing = index.get(decision.matched_entity_id)
                if existing is not None:
                    merged = existing.merged_with(entity)
                    index.replace(merged)
                    resolved.append(merged)
                    continue

            canonical = entity.canonical_name or canonicalize(entity.name, entity.type)
            indexed = index.add(
                entity.model_copy(update={"canonical_name": canonical, "resolved": True})
            )
            resolved.append(indexed)

        return resolved, decisions

    @staticmethod
    def _compare_identifiers(left: Entity, right: Entity) -> tuple[set[str], set[str]]:
        """Identifier types that agree and that conflict between two entities."""
        shared: set[str] = set()
        conflicting: set[str] = set()

        left_by_type = {identifier.type: identifier.value for identifier in left.identifiers}
        right_by_type = {identifier.type: identifier.value for identifier in right.identifiers}

        for identifier_type, left_value in left_by_type.items():
            right_value = right_by_type.get(identifier_type)
            if right_value is None:
                continue
            if right_value == left_value:
                shared.add(str(identifier_type))
            elif identifier_type in AUTHORITATIVE_IDENTIFIERS or identifier_type is (
                IdentifierType.TICKER
            ):
                conflicting.add(str(identifier_type))

        return shared, conflicting

    @staticmethod
    def _alias_matches(canonical: str, candidate: Entity) -> bool:
        return any(canonicalize(alias, candidate.type) == canonical for alias in candidate.aliases)

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        # token_sort_ratio so word order does not matter ("Apple Computer" vs
        # "Computer Apple"), which is common across data feeds.
        return float(fuzz.token_sort_ratio(left, right))

    @staticmethod
    def _token_overlap(left: str, right: str) -> float:
        left_tokens, right_tokens = name_tokens(left), name_tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))


__all__ = [
    "AUTHORITATIVE_IDENTIFIERS",
    "EntityIndex",
    "EntityResolver",
    "MatchDecision",
    "MatchReason",
    "ResolutionConfig",
    "canonicalize",
]
