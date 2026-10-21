from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from coldchain import __version__
from coldchain.api.container import Container, build_container
from coldchain.api.metrics import HTTP_LATENCY, HTTP_REQUESTS
from coldchain.api.routes import chat, ops
from coldchain.api.schemas import Health
from coldchain.config import Settings, get_settings
from coldchain.db.session import ping
from coldchain.logging import configure_logging, get_logger

log = get_logger(__name__)


def create_app(
    settings: Settings | None = None,
    container_factory: Callable[[Settings], Container] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = (container_factory or build_container)(settings)
        log.info("startup", env=settings.env, llm=settings.llm_provider, model=settings.llm_model)
        yield
        app.state.container.close()

    app = FastAPI(
        title="Cold-Chain Ops Assistant",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if settings.env != "prod" else None,
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=rid)
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            route = getattr(request.scope.get("route"), "path", "unmatched")
            HTTP_REQUESTS.labels(request.method, route, status).inc()
            HTTP_LATENCY.labels(request.method, route).observe(time.perf_counter() - t0)
            structlog.contextvars.clear_contextvars()

    @app.get("/healthz", response_model=Health, tags=["meta"])
    def healthz() -> Health:
        return Health(status="ok", version=__version__)

    @app.get("/readyz", response_model=Health, tags=["meta"])
    def readyz(response: Response, request: Request) -> Health:
        db_ok = ping(request.app.state.container.engine)
        if not db_ok:
            response.status_code = 503
        return Health(status="ok" if db_ok else "degraded", version=__version__, database=db_ok)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.include_router(ops.router)
    app.include_router(chat.router)
    return app
