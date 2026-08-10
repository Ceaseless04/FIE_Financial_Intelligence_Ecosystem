"""Entity resolution: the decision ladder, and the cases it must refuse."""

from __future__ import annotations

import pytest
from marketmind_fixtures import make_entity

from marketmind.domain.entities import EntityType, IdentifierType
from marketmind.resolution.normalization import (
    blocking_key,
    normalize_company_name,
    normalize_person_name,
)
from marketmind.resolution.resolver import (
    EntityIndex,
    EntityResolver,
    MatchReason,
    ResolutionConfig,
    canonicalize,
)

pytestmark = pytest.mark.unit


class TestNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Acme Robotics Corporation", "acme robotics"),
            ("Acme Robotics Corp.", "acme robotics"),
            ("Acme Robotics, Inc.", "acme robotics"),
            ("ACME ROBOTICS CO., LTD.", "acme robotics"),
            # "Holdings" is a legal suffix too, so a holdco and its operating
            # company canonicalize together — which is the intended behaviour.
            ("Acme Robotics Holdings PLC", "acme robotics"),
        ],
    )
    def test_legal_suffixes_are_stripped(self, raw: str, expected: str) -> None:
        assert normalize_company_name(raw) == expected

    def test_stripping_never_empties_a_name(self) -> None:
        """A company literally called "Group" must keep something to match on."""
        assert normalize_company_name("Group") != ""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Marchetti, Sofia", "sofia marchetti"),
            ("Ms. Sofia Marchetti", "sofia marchetti"),
            ("Sofia A. Marchetti", "sofia marchetti"),
            ("SOFIA MARCHETTI", "sofia marchetti"),
        ],
    )
    def test_person_names_normalize_to_one_form(self, raw: str, expected: str) -> None:
        assert normalize_person_name(raw) == expected

    def test_blocking_key_groups_variants_together(self) -> None:
        """Blocking must not separate names that resolution should compare."""
        assert blocking_key(normalize_company_name("Acme Robotics Corporation")) == blocking_key(
            normalize_company_name("Acme Robotics Corp")
        )

    def test_canonicalize_dispatches_on_entity_type(self) -> None:
        assert canonicalize("Marchetti, Sofia", EntityType.EXECUTIVE) == "sofia marchetti"
        assert canonicalize("Acme Robotics Inc", EntityType.COMPANY) == "acme robotics"


class TestResolutionLadder:
    def test_shared_identifier_decides_a_match(self) -> None:
        index = EntityIndex()
        index.add(
            make_entity("Acme Robotics Corporation", identifiers=[(IdentifierType.CIK, "123456")])
        )
        incoming = make_entity("A.R.C. Holdings", identifiers=[(IdentifierType.CIK, "0000123456")])

        decision = EntityResolver().resolve(incoming, index)

        assert decision.matched
        assert decision.reason is MatchReason.IDENTIFIER_MATCH
        assert decision.confidence == 1.0
        assert decision.is_automatic

    def test_conflicting_identifier_blocks_a_match_despite_identical_names(self) -> None:
        """Two 'Acme Corp' with different CIKs are two companies."""
        index = EntityIndex()
        index.add(make_entity("Acme Corp", identifiers=[(IdentifierType.CIK, "111")]))
        incoming = make_entity("Acme Corp", identifiers=[(IdentifierType.CIK, "222")])

        decision = EntityResolver().resolve(incoming, index)

        assert not decision.matched

    def test_partial_identifier_agreement_refuses_to_decide(self) -> None:
        index = EntityIndex()
        index.add(
            make_entity(
                "Acme Corp",
                identifiers=[(IdentifierType.CIK, "111"), (IdentifierType.TICKER, "ACME")],
            )
        )
        incoming = make_entity(
            "Acme Corp",
            identifiers=[(IdentifierType.CIK, "111"), (IdentifierType.TICKER, "ACMX")],
        )

        decision = EntityResolver().resolve(incoming, index)

        assert not decision.matched
        assert decision.reason is MatchReason.IDENTIFIER_CONFLICT
        assert "conflicting" in decision.evidence

    def test_identical_canonical_name_matches(self) -> None:
        index = EntityIndex()
        index.add(make_entity("Acme Robotics Corporation"))

        decision = EntityResolver().resolve(make_entity("Acme Robotics, Inc."), index)

        assert decision.matched
        assert decision.reason is MatchReason.EXACT_NAME_MATCH

    def test_alias_matches(self) -> None:
        index = EntityIndex()
        index.add(make_entity("Acme Robotics Corporation", aliases=["Acme Automation"]))

        decision = EntityResolver().resolve(make_entity("Acme Automation Inc"), index)

        assert decision.matched
        assert decision.reason in (MatchReason.ALIAS_MATCH, MatchReason.EXACT_NAME_MATCH)

    def test_short_similar_names_do_not_merge(self) -> None:
        """The token-overlap guard is what stops 'Meta' matching 'Metro'."""
        index = EntityIndex()
        index.add(make_entity("Metro Holdings"))

        decision = EntityResolver().resolve(make_entity("Meta Holdings"), index)

        assert not decision.matched

    def test_unknown_entity_produces_no_match(self) -> None:
        decision = EntityResolver().resolve(make_entity("Wholly Unrelated"), EntityIndex())
        assert not decision.matched
        assert decision.reason is MatchReason.NO_MATCH
        assert decision.evidence == "no candidates"

    def test_every_decision_carries_evidence(self) -> None:
        """A merge nobody can explain is a merge nobody can undo."""
        index = EntityIndex()
        index.add(make_entity("Acme Robotics Corporation"))
        for incoming in (
            make_entity("Acme Robotics Inc"),
            make_entity("Zenith Automation"),
        ):
            assert EntityResolver().resolve(incoming, index).evidence

    def test_resolution_is_reproducible(self) -> None:
        index = EntityIndex()
        index.add(make_entity("Acme Robotics Corporation"))
        incoming = make_entity("Acme Robotics Corp")

        first = EntityResolver().resolve(incoming, index)
        second = EntityResolver().resolve(incoming, index)

        assert first.model_dump() == second.model_dump()


class TestBatchResolution:
    def test_variants_within_one_batch_collapse_to_one_entity(self) -> None:
        resolver = EntityResolver()
        index = EntityIndex()
        entities = [
            make_entity("Acme Robotics Corporation"),
            make_entity("Acme Robotics Corp"),
            make_entity("Zenith Automation Inc"),
        ]

        resolved, decisions = resolver.resolve_batch(entities, index)

        assert len(resolved) == 3
        assert resolved[0].id == resolved[1].id
        assert resolved[2].id != resolved[0].id
        assert decisions[1].matched
        assert len(index) == 2

    def test_merged_entity_keeps_the_variant_as_an_alias(self) -> None:
        resolver = EntityResolver()
        index = EntityIndex()
        resolved, _ = resolver.resolve_batch(
            [make_entity("Acme Robotics Corporation"), make_entity("Acme Robotics Corp")],
            index,
        )
        assert "Acme Robotics Corp" in resolved[1].aliases

    def test_resolved_entities_are_marked_resolved(self) -> None:
        resolved, _ = EntityResolver().resolve_batch([make_entity("Acme")], EntityIndex())
        assert resolved[0].resolved

    def test_canonical_name_is_assigned(self) -> None:
        resolved, _ = EntityResolver().resolve_batch(
            [make_entity("Acme Robotics Corporation")], EntityIndex()
        )
        assert resolved[0].canonical_name == "acme robotics"


class TestThresholds:
    def test_a_stricter_threshold_refuses_a_looser_match(self) -> None:
        index = EntityIndex()
        index.add(make_entity("Acme Robotics Systems"))
        incoming = make_entity("Acme Robotics System")

        lenient = EntityResolver(ResolutionConfig(fuzzy_threshold=80.0)).resolve(incoming, index)
        strict = EntityResolver(ResolutionConfig(fuzzy_threshold=99.5)).resolve(incoming, index)

        assert lenient.matched
        assert lenient.reason is MatchReason.FUZZY_NAME_MATCH
        assert not strict.matched

    def test_fuzzy_confidence_never_reaches_identifier_confidence(self) -> None:
        """A name match must never look as certain as a CIK match."""
        index = EntityIndex()
        index.add(make_entity("Acme Robotics Systems"))
        decision = EntityResolver(ResolutionConfig(fuzzy_threshold=80.0)).resolve(
            make_entity("Acme Robotics System"), index
        )
        assert decision.matched
        assert decision.confidence < 1.0

    def test_token_overlap_guard_outranks_a_high_similarity_score(self) -> None:
        """A 95-point score on names sharing one token in three is not identity."""
        index = EntityIndex()
        index.add(make_entity("Acme Robotics Systems"))

        decision = EntityResolver(ResolutionConfig(fuzzy_threshold=80.0)).resolve(
            make_entity("Acme Robotic System"), index
        )

        assert not decision.matched
        assert "token overlap" in decision.evidence
