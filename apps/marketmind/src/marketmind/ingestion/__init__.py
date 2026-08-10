"""Ingestion: document in, resolved graph structure out."""

from marketmind.ingestion.chunking import (
    ChunkingConfig,
    chunk_document,
    verify_chunk_offsets,
)
from marketmind.ingestion.extraction import (
    ExtractionOutcome,
    ExtractionResult,
    ExtractionService,
)
from marketmind.ingestion.metadata import extract_metadata
from marketmind.ingestion.pipeline import IngestionPipeline, IngestionReport

__all__ = [
    "ChunkingConfig",
    "ExtractionOutcome",
    "ExtractionResult",
    "ExtractionService",
    "IngestionPipeline",
    "IngestionReport",
    "chunk_document",
    "extract_metadata",
    "verify_chunk_offsets",
]
