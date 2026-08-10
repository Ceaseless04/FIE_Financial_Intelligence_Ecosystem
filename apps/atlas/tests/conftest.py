"""Pytest fixtures for the Atlas suites.

The builders live in :mod:`atlas_fixtures`, imported by bare module name from
this directory. Only pytest fixtures belong here: a helper defined in a
``conftest`` and imported as ``from conftest import ...`` resolves through the
process-wide ``sys.modules``, so the first app to be collected wins and every
other app's tests error at setup.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from atlas_fixtures import statements

from atlas.domain.statements import FinancialStatements


@pytest.fixture
def acme_statements() -> FinancialStatements:
    return statements()
