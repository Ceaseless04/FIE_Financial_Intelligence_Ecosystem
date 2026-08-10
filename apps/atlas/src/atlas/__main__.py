"""Run the Atlas API.

    python -m atlas

Exists so the host and port come from configuration rather than being repeated
in a container CMD, a compose file, and a runbook — three places that drift.
"""

from __future__ import annotations

import uvicorn

from atlas.config import AtlasSettings


def main() -> None:
    settings = AtlasSettings()
    uvicorn.run(
        "atlas.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        # Logging is configured by the application itself (structlog, JSON in
        # production); uvicorn's own config would fight it.
        log_config=None,
    )


if __name__ == "__main__":
    main()
