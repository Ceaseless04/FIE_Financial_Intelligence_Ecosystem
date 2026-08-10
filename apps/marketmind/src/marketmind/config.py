"""MarketMind configuration.

Everything resolves from ``MARKETMIND_``-prefixed environment variables. The
shared infrastructure settings (Postgres, Neo4j, Redis, AI providers) come from
their own packages, so this class holds only what is specific to the knowledge
graph.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from fie_common.config import Environment, FIEBaseSettings
from fie_common.errors import ConfigurationError
from marketmind.ingestion.chunking import ChunkingConfig
from marketmind.resolution.resolver import ResolutionConfig
from marketmind.retrieval.graphrag import GraphRAGConfig
from marketmind.storage.models import EMBEDDING_DIMENSIONS


class MarketMindSettings(FIEBaseSettings):
    """MarketMind service configuration (``MARKETMIND_*``)."""

    model_config = SettingsConfigDict(
        env_prefix="MARKETMIND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    service_name: str = "marketmind"
    #: Loopback by default. Binding every interface is a deployment decision,
    #: not a default: the container image opts in explicitly by setting
    #: MARKETMIND_API_HOST, so a developer running the service locally does not
    #: silently expose it to their network.
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8001, ge=1, le=65535)

    # -- embeddings ----------------------------------------------------------
    embedding_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    #: Must equal the pgvector column width; see ``validate_for``.
    embedding_dimensions: int = Field(default=EMBEDDING_DIMENSIONS, ge=1)
    embedding_timeout_seconds: float = Field(default=60.0, gt=0)
    #: Texts embedded per request. Ollama holds the whole batch in memory.
    embedding_batch_size: int = Field(default=32, ge=1, le=512)

    # -- chunking ------------------------------------------------------------
    chunk_max_chars: int = Field(default=1200, ge=50)
    chunk_overlap_chars: int = Field(default=150, ge=0)
    chunk_min_chars: int = Field(default=100, ge=0)

    # -- extraction ----------------------------------------------------------
    #: Chunks per extraction call. Larger batches cost fewer calls but give the
    #: model more room to attribute a fact to the wrong chunk.
    extraction_chunks_per_call: int = Field(default=6, ge=1, le=40)
    extraction_max_tokens: int = Field(default=4096, ge=256)
    #: Cap on chunks extracted from per document. A 300-page 10-K would
    #: otherwise turn one ingestion into hundreds of model calls.
    extraction_max_chunks: int = Field(default=60, ge=1)

    # -- resolution ----------------------------------------------------------
    resolution_fuzzy_threshold: float = Field(default=92.0, ge=0, le=100)
    resolution_minimum_candidate_score: float = Field(default=80.0, ge=0, le=100)
    resolution_minimum_token_overlap: float = Field(default=0.5, ge=0, le=1)

    # -- retrieval -----------------------------------------------------------
    retrieval_top_k_chunks: int = Field(default=6, ge=1, le=100)
    retrieval_graph_depth: int = Field(default=1, ge=1, le=3)
    retrieval_max_neighbours: int = Field(default=20, ge=1, le=500)
    retrieval_max_seed_entities: int = Field(default=5, ge=1, le=50)
    retrieval_minimum_similarity: float = Field(default=0.15, ge=0, le=1)
    retrieval_max_answer_tokens: int = Field(default=2048, ge=256)

    # -- graph ---------------------------------------------------------------
    #: APOC ships with the dev Neo4j image but is absent from some managed
    #: offerings, so it is opt-in rather than assumed.
    graph_use_apoc: bool = False
    graph_visualization_max_nodes: int = Field(default=250, ge=1, le=2000)

    # -- events --------------------------------------------------------------
    events_consumer_group: str = "marketmind"
    publish_events: bool = True

    def chunking(self) -> ChunkingConfig:
        return ChunkingConfig(
            max_chars=self.chunk_max_chars,
            overlap_chars=self.chunk_overlap_chars,
            min_chars=self.chunk_min_chars,
        )

    def resolution(self) -> ResolutionConfig:
        return ResolutionConfig(
            fuzzy_threshold=self.resolution_fuzzy_threshold,
            minimum_candidate_score=self.resolution_minimum_candidate_score,
            minimum_token_overlap=self.resolution_minimum_token_overlap,
        )

    def graphrag(self) -> GraphRAGConfig:
        return GraphRAGConfig(
            top_k_chunks=self.retrieval_top_k_chunks,
            graph_depth=self.retrieval_graph_depth,
            max_neighbours=self.retrieval_max_neighbours,
            max_seed_entities=self.retrieval_max_seed_entities,
            minimum_similarity=self.retrieval_minimum_similarity,
            max_answer_tokens=self.retrieval_max_answer_tokens,
        )

    def validate_for(self, environment: Environment) -> None:
        """Fail at startup rather than at the first write.

        The embedding width is the one setting that cannot be wrong quietly:
        pgvector fixes the column width in a migration, so a mismatched model
        fails every insert with an opaque dimension error long after the
        misconfiguration happened.

        Raises:
            ConfigurationError: on a dimension mismatch, or on a chunk overlap
                that would make the chunker loop.
        """
        if self.embedding_dimensions != EMBEDDING_DIMENSIONS:
            raise ConfigurationError(
                "MARKETMIND_EMBEDDING_DIMENSIONS does not match the pgvector column width; "
                "changing the embedding model requires a migration",
                details={
                    "configured": self.embedding_dimensions,
                    "column_width": EMBEDDING_DIMENSIONS,
                    "model": self.embedding_model,
                },
            )
        if self.chunk_overlap_chars >= self.chunk_max_chars:
            raise ConfigurationError(
                "MARKETMIND_CHUNK_OVERLAP_CHARS must be smaller than the chunk size",
                details={
                    "overlap": self.chunk_overlap_chars,
                    "max_chars": self.chunk_max_chars,
                },
            )
        if environment.is_production and self.embedding_base_url.startswith(
            ("http://localhost", "http://127.0.0.1")
        ):
            raise ConfigurationError(
                "MARKETMIND_EMBEDDING_BASE_URL still points at localhost in production",
                details={"base_url": self.embedding_base_url},
            )


__all__ = ["MarketMindSettings"]
