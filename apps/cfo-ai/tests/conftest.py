"""Pytest fixtures for the CFO.ai suites.

Builders live in :mod:`cfo_fixtures`; only pytest fixtures belong here. See that
module for why the name is app-specific.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from cfo_fixtures import actuals, budget

from cfo_ai.domain.plan import Plan


@pytest.fixture
def fy26_budget() -> Plan:
    return budget()


@pytest.fixture
def fy26_actuals() -> Plan:
    return actuals()
