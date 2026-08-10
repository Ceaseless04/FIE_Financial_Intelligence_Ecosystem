"""Root pytest configuration.

Unit tests must not read ambient configuration. A test that asserts a settings
*default* while the surrounding shell exports that setting is not testing a
default — it is testing whatever the environment happens to say.

Configuration reaches a settings class by two routes, and both have to be shut
off or the suite's result depends on who is running it:

1. **Environment variables.** CI exports `FIE_ENVIRONMENT=test` for every job,
   which is what made `test_defaults_are_development_safe` fail there and pass
   locally. Several sibling tests passed only because the other values CI
   exports happen to equal the declared defaults — coincidence, not coverage.
2. **The `.env` file.** Every settings class declares `env_file=".env"`, and
   `.env.example` sets `FIE_LOG_LEVEL=DEBUG` and `FIE_DEBUG=true`. A developer
   who follows the README's `cp .env.example .env` therefore gets three failing
   unit tests on a clean checkout — the mirror image of the CI failure.

Clearing the variables handles (1). For (2), the file is resolved relative to
the working directory, so unit tests run from an empty temporary one.

Integration, API, and end-to-end tests are deliberately left alone: they need
`FIE_POSTGRES_HOST` and friends to find the infrastructure they run against.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

#: Configuration namespaces owned by this repository. Anything with these
#: prefixes is service configuration, and a unit test that depends on one is
#: reading its environment rather than exercising the code.
_CONFIG_PREFIXES = ("FIE_", "MARKETMIND_")


@pytest.fixture(autouse=True)
def _isolate_unit_tests_from_the_environment(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cut every unit test off from ambient service configuration.

    Applied through ``monkeypatch`` so both the environment and the working
    directory are restored afterwards, and scoped by marker so integration tests
    keep the connection details they need. A unit test that genuinely wants a
    variable set can still call ``monkeypatch.setenv`` in its body — that runs
    after this fixture.
    """
    if "unit" not in request.keywords:
        return

    for name in list(os.environ):
        if name.startswith(_CONFIG_PREFIXES):
            monkeypatch.delenv(name, raising=False)

    # `env_file=".env"` resolves against the working directory, so an empty one
    # is what makes a developer's local .env invisible here. Deleting the
    # variables alone would not do it.
    monkeypatch.chdir(tmp_path)
