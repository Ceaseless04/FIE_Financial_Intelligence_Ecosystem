"""Chunking determinism, offset integrity, and regex metadata extraction."""

from __future__ import annotations

from itertools import pairwise

import pytest
from fixtures.documents import acme_10k, northwind_10k, unstructured_note
from pydantic import ValidationError as PydanticValidationError

from fie_common.errors import ValidationError
from marketmind.domain.documents import Document, DocumentType
from marketmind.ingestion.chunking import (
    ChunkingConfig,
    chunk_document,
    verify_chunk_offsets,
)
from marketmind.ingestion.metadata import extract_metadata

pytestmark = pytest.mark.unit


class TestChunking:
    def test_chunking_is_deterministic(self) -> None:
        """Same input, same chunks — embeddings and citations depend on it."""
        document = acme_10k()
        first = chunk_document(document)
        second = chunk_document(document)

        assert [(c.index, c.start_char, c.end_char, c.text) for c in first] == [
            (c.index, c.start_char, c.end_char, c.text) for c in second
        ]

    def test_chunk_ids_are_stable_across_re_ingestion(self) -> None:
        """Re-ingestion must not orphan the mention rows built alongside chunks."""
        document = acme_10k()
        assert [c.id for c in chunk_document(document)] == [c.id for c in chunk_document(document)]

    def test_chunk_ids_differ_between_documents(self) -> None:
        first = [c.id for c in chunk_document(acme_10k())]
        second = [c.id for c in chunk_document(northwind_10k())]
        assert not set(first) & set(second)

    def test_every_chunk_offset_resolves_to_its_text(self) -> None:
        document = acme_10k()
        chunks = chunk_document(document)
        assert verify_chunk_offsets(document, chunks)
        for chunk in chunks:
            assert document.content[chunk.start_char : chunk.end_char] == chunk.text

    def test_offsets_hold_for_unstructured_text(self) -> None:
        """The fallback path splits without headings and must stay exact."""
        document = unstructured_note()
        chunks = chunk_document(document)
        assert len(chunks) > 1
        assert verify_chunk_offsets(document, chunks)

    def test_chunks_are_indexed_in_order(self) -> None:
        chunks = chunk_document(acme_10k())
        assert [chunk.index for chunk in chunks] == list(range(len(chunks)))

    def test_headings_become_sections(self) -> None:
        chunks = chunk_document(acme_10k())
        sections = {chunk.section for chunk in chunks if chunk.section}
        assert any("ITEM 1A" in section for section in sections)
        assert any("RISK FACTORS" in section.upper() for section in sections)

    def test_chunks_respect_the_size_limit(self) -> None:
        config = ChunkingConfig(max_chars=300, overlap_chars=40, min_chars=50)
        chunks = chunk_document(acme_10k(), config)
        # A chunk may absorb an undersized tail, so allow one overlap of slack.
        assert all(len(chunk.text) <= config.max_chars + config.overlap_chars for chunk in chunks)

    def test_a_blank_document_cannot_be_constructed(self) -> None:
        with pytest.raises(PydanticValidationError):
            Document(type=DocumentType.NEWS_ARTICLE, title="Empty", content="   \n\t  ")

    def test_chunker_refuses_blank_content_that_skipped_validation(self) -> None:
        """model_construct bypasses validation, so the chunker keeps its guard."""
        document = Document.model_construct(
            id="doc-blank", type=DocumentType.NEWS_ARTICLE, title="Empty", content="   \n\t  "
        )
        with pytest.raises(ValidationError, match="no content"):
            chunk_document(document)

    def test_short_document_still_produces_one_chunk(self) -> None:
        """A brief news item must not vanish for being under min_chars."""
        document = Document(
            type=DocumentType.NEWS_ARTICLE, title="Brief", content="Acme names a new CEO."
        )
        chunks = chunk_document(document)
        assert len(chunks) == 1
        assert verify_chunk_offsets(document, chunks)

    @pytest.mark.parametrize(
        ("max_chars", "overlap", "min_chars"),
        [(10, 5, 0), (200, 200, 50), (200, 20, 500)],
    )
    def test_invalid_configuration_is_rejected(
        self, max_chars: int, overlap: int, min_chars: int
    ) -> None:
        with pytest.raises(ValueError):
            ChunkingConfig(max_chars=max_chars, overlap_chars=overlap, min_chars=min_chars)

    def test_overlap_carries_context_across_a_boundary(self) -> None:
        config = ChunkingConfig(max_chars=200, overlap_chars=60, min_chars=40)
        chunks = chunk_document(acme_10k(), config)
        same_section = [(a, b) for a, b in pairwise(chunks) if a.section == b.section]
        assert same_section, "expected at least one within-section split"
        assert any(b.start_char < a.end_char for a, b in same_section)


class TestMetadataExtraction:
    def test_exchange_ticker_is_recovered(self) -> None:
        metadata = extract_metadata(acme_10k())
        assert "ACME" in metadata.tickers

    def test_cik_padding_is_stripped(self) -> None:
        metadata = extract_metadata(acme_10k())
        assert "123456" in metadata.ciks

    def test_form_type_and_period_are_recovered(self) -> None:
        metadata = extract_metadata(acme_10k())
        assert metadata.form_type == "10-K"
        assert metadata.fiscal_year == 2025
        assert metadata.period_ended is not None
        assert metadata.period_ended.year == 2025

    def test_metadata_converts_to_identifiers(self) -> None:
        identifiers = extract_metadata(northwind_10k()).identifiers
        keys = {identifier.key for identifier in identifiers}
        assert "ticker:NWC" in keys
        assert "cik:987654" in keys

    def test_feed_supplied_ticker_takes_precedence(self) -> None:
        """The feed knows the ticker; the regex is guessing."""
        document = Document(
            type=DocumentType.NEWS_ARTICLE,
            title="Coverage note",
            content="Shares of THE and AND moved today. (NASDAQ: WRONG)",
            metadata={"ticker": "RIGHT"},
        )
        metadata = extract_metadata(document)
        assert metadata.tickers[0] == "RIGHT"

    def test_common_words_are_not_mistaken_for_tickers(self) -> None:
        document = Document(
            type=DocumentType.NEWS_ARTICLE,
            title="All caps",
            content="THE COMPANY AND ITS CEO AND CFO DISCUSSED GAAP RESULTS.",
        )
        assert extract_metadata(document).tickers == []

    def test_a_four_digit_dollar_figure_is_not_a_fiscal_year(self) -> None:
        document = Document(
            type=DocumentType.NEWS_ARTICLE,
            title="Numbers",
            content="Revenue reached 4126 units across the period.",
        )
        assert extract_metadata(document).fiscal_year is None

    def test_extraction_never_calls_a_model(self) -> None:
        """Deterministic by construction: no provider is reachable from here."""
        import marketmind.ingestion.metadata as module

        source = module.__file__
        assert source is not None
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        assert "AIRouter" not in text
        assert "CompletionRequest" not in text
