"""Run the CFO.ai API.

    python -m cfo_ai

Exists so the host and port come from configuration rather than being repeated
in a container CMD, a compose file, and a runbook — three places that drift.
"""

from __future__ import annotations

import uvicorn

from cfo_ai.config import CFOSettings


def main() -> None:
    settings = CFOSettings()
    uvicorn.run(
        "cfo_ai.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        # Logging is configured by the application itself (structlog, JSON in
        # production); uvicorn's own config would fight it.
        log_config=None,
    )


if __name__ == "__main__":
    main()
