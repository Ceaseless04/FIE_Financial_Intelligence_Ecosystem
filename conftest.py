"""Root pytest configuration.

Unit tests must not read ambient configuration. A test that asserts a settings
*default* while the surrounding shell exports that setting is not testing a
default — it is testing whatever the environment happens to say, which is how a
suite passes locally and fails in CI (where `FIE_ENVIRONMENT=test` is set for
every job).

Several settings tests were passing only because the values CI exports happen to
equal the declared defaults. That is coincidence, not coverage, so the isolation
is applied to every unit test rather than patched into the one that noticed.

Integration, API, and end-to-end tests are deliberately left alone: they need
`FIE_POSTGRES_HOST` and friends to find the infrastructure they run against.
"""

from __future__ import annotations

import os

import pytest

#: Configuration namespaces owned by this repository. Anything with these
#: prefixes is service configuration, and a unit test that depends on one is
#: reading its environment rather than exercising the code.
_CONFIG_PREFIXES = ("FIE_", "MARKETMIND_")


@pytest.fixture(autouse=True)
def _isolate_unit_tests_from_the_environment(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strip service configuration from the environment of every unit test.

    Applied through ``monkeypatch`` so the original environment is restored
    afterwards, and scoped by marker so integration tests keep the connection
    details they need. A unit test that genuinely wants a variable set can still
    call ``monkeypatch.setenv`` in its body — that runs after this fixture.
    """
    if "unit" not in request.keywords:
        return

    for name in list(os.environ):
        if name.startswith(_CONFIG_PREFIXES):
            monkeypatch.delenv(name, raising=False)
