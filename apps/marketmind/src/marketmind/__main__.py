"""Run the MarketMind API.

    python -m marketmind

Exists so the host and port come from configuration rather than being repeated
in a container CMD, a compose file, and a runbook — three places that drift.
"""

from __future__ import annotations

import uvicorn

from marketmind.config import MarketMindSettings


def main() -> None:
    settings = MarketMindSettings()
    uvicorn.run(
        "marketmind.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        # Logging is configured by the application itself (structlog, JSON in
        # production); uvicorn's own config would fight it.
        log_config=None,
    )


if __name__ == "__main__":
    main()
