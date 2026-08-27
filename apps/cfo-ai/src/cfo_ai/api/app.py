"""FastAPI application factory.

The factory takes an optional pre-built container so tests can inject fakes and
exercise the real routing, validation, authorization, and error translation. An
API test that stubs out the app itself proves nothing about the app.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from cfo_ai import __version__
from cfo_ai.api.dependencies import ServiceContainer
from cfo_ai.api.routers import commentary, health, plans, variance
from cfo_ai.config import CFOSettings
from fie_common.config import CoreSettings
from fie_common.errors import FIEError, ValidationError
from fie_common.utils import new_id
from fie_observability.context import current_context, request_context
from fie_observability.logging import configure_logging, get_logger
from fie_observability.tracing import configure_tracing
from fie_schemas.envelope import ErrorDetail

logger = get_logger(__name__)

#: Header carrying a correlation id across service boundaries.
CORRELATION_HEADER = "X-Correlation-ID"


def create_app(
    *,
    container: ServiceContainer | None = None,
    settings: CFOSettings | None = None,
    core: CoreSettings | None = None,
    configure_observability: bool = True,
) -> FastAPI:
    """Build the CFO.ai API.

    Args:
        container: Pre-built dependencies. When omitted, the container is
            constructed from the environment during startup, so importing this
            module never requires a running database.
    """
    settings = settings or CFOSettings()
    # Logs, traces, and metrics identify a service by CoreSettings.service_name.
    # CFO.ai is the authority on its own name, so it is stamped here rather than
    # leaving every deployment to keep FIE_SERVICE_NAME in sync with the app it
    # happens to be running.
    core = (core or CoreSettings()).model_copy(update={"service_name": settings.service_name})

    if configure_observability:
        configure_logging(core)
        configure_tracing(core)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_container = container is None
        active = container or ServiceContainer.build(settings=settings, core=core)
        app.state.container = active
        logger.info(
            "cfo_ai_started",
            environment=str(core.environment),
            material_percent=str(active.settings.material_percent),
        )
        try:
            yield
        finally:
            # Only tear down what this app created; an injected container
            # belongs to the caller and may outlive the app.
            if owns_container:
                await active.aclose()
            logger.info("cfo_ai_stopped")

    app = FastAPI(
        title="CFO.ai",
        version=__version__,
        description=(
            "Financial planning and analysis: budgets, actuals, variance, "
            "forecasting, and scenarios. Every figure is computed "
            "deterministically, and so is the *direction* of every variance — "
            "a cost below plan is favourable, revenue below plan is not, and "
            "the chart of accounts decides which. The language model explains "
            "those figures and is checked against both them and their "
            "directions before anything is published."
        ),
        lifespan=lifespan,
        # The interactive docs are useful in development and are an information
        # leak in production, where the schema is published deliberately or not
        # at all.
        docs_url=None if core.environment.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if core.environment.is_production else "/openapi.json",
    )

    _install_middleware(app)
    _install_error_handlers(app)

    app.include_router(health.router)
    app.include_router(plans.router)
    app.include_router(variance.router)
    app.include_router(commentary.router)

    return app


def _install_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def correlation_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[JSONResponse]]
    ) -> JSONResponse:
        """Bind a correlation id to everything emitted while handling a request.

        An inbound id is honoured so a call that began elsewhere keeps one id
        across every service it touches — including the MarketMind lookup CFO.ai
        makes on the way — otherwise one is minted here.
        """
        correlation_id = request.headers.get(CORRELATION_HEADER) or new_id("corr")
        started = time.perf_counter()

        # Also stashed on the request because the ambient context is scoped to
        # the block below: when a handler raises, the context is already unwound
        # by the time Starlette's outermost error middleware runs the 500
        # handler — which is exactly the response an operator needs the id on.
        request.state.correlation_id = correlation_id

        with request_context(
            correlation_id=correlation_id,
            request_id=new_id("req"),
            service_name="cfo_ai",
            source_app="cfo_ai",
        ):
            response = await call_next(request)

        response.headers[CORRELATION_HEADER] = correlation_id
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return response


def _correlation_id(request: Request) -> str | None:
    """The active correlation id, from the request or the ambient context.

    The request wins because it survives the context being unwound by an
    exception on its way to the outermost error middleware.
    """
    stashed = getattr(request.state, "correlation_id", None)
    if stashed:
        return str(stashed)
    return current_context().correlation_id or None


def _install_error_handlers(app: FastAPI) -> None:
    """Translate platform errors into the shared error envelope.

    Every error leaves through one of these handlers, so a caller sees the same
    body shape whether the failure came from validation, authorization, or a
    dependency — and an unexpected exception never leaks a stack trace.
    """

    @app.exception_handler(FIEError)
    async def fie_error_handler(request: Request, error: FIEError) -> JSONResponse:
        correlation_id = _correlation_id(request)
        log = logger.warning if error.http_status < 500 else logger.error
        log(
            "request_failed",
            error_code=error.code,
            status_code=error.http_status,
            path=request.url.path,
            correlation_id=correlation_id,
        )
        return JSONResponse(
            status_code=error.http_status,
            content=ErrorDetail.from_error(error, correlation_id=correlation_id).model_dump(
                mode="json"
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, error: RequestValidationError) -> JSONResponse:
        translated = ValidationError(
            "request validation failed",
            details={"errors": error.errors()[:10]},
        )
        return JSONResponse(
            status_code=translated.http_status,
            content=ErrorDetail.from_error(
                translated, correlation_id=_correlation_id(request)
            ).model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, error: Exception) -> JSONResponse:
        correlation_id = _correlation_id(request)
        logger.exception(
            "unhandled_error",
            path=request.url.path,
            error_type=type(error).__name__,
            correlation_id=correlation_id,
        )
        # The message is deliberately opaque; the correlation id is how an
        # operator finds the detail in the logs without it being served to the
        # caller.
        return JSONResponse(
            status_code=500,
            content=ErrorDetail(
                code="internal_error",
                message="an unexpected error occurred",
                retryable=False,
                correlation_id=correlation_id,
            ).model_dump(mode="json"),
        )


__all__ = ["CORRELATION_HEADER", "create_app"]
