"""Pytest fixtures for the MarketMind suites.

The builders live in :mod:`marketmind_fixtures`, imported by bare module name
from this directory. Only pytest fixtures belong here: a helper defined in a
``conftest`` and imported as ``from conftest import ...`` resolves through the
process-wide ``sys.modules``, so the first app to be collected wins and every
other app's tests error at setup.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Also puts the `fixtures` package on the path, so tests can say
# `from fixtures.documents import acme_10k` under importlib import mode.
sys.path.insert(0, str(Path(__file__).parent))

from marketmind_fixtures import make_source

from fie_schemas.provenance import SourceReference
from fie_testing import DeterministicEmbeddingProvider


@pytest.fixture
def embeddings() -> DeterministicEmbeddingProvider:
    """Deterministic embeddings with real vector geometry."""
    return DeterministicEmbeddingProvider(dimensions=64)


@pytest.fixture
def source() -> SourceReference:
    return make_source()
