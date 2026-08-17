from __future__ import annotations

import logging

from fastapi import FastAPI

from app.config import settings
from app.routes.health import router as health_router
from app.routes.query import router as query_router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="astra",
        description="RAG microservice for PASTI PINTAR",
        version="0.1.0",
        docs_url=None,  # Internal service — no public docs
        redoc_url=None,
    )

    # Include routers
    app.include_router(health_router)
    app.include_router(query_router)

    @app.on_event("startup")
    async def on_startup():
        # Log config presence (not values)
        _log_config_presence()
        logger.info("astra service starting on port %d", settings.APP_PORT)

    return app


def _log_config_presence():
    checks = {
        "DATABASE_URL": bool(settings.DATABASE_URL),
        "OPENAI_API_KEY": bool(settings.OPENAI_API_KEY),
        "OPENROUTER_API_KEY": bool(settings.OPENROUTER_API_KEY),
        "INTERNAL_API_KEY": bool(settings.INTERNAL_API_KEY),
        "LANGFUSE": bool(settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY),
    }
    for name, present in checks.items():
        logger.info("Config %s: %s", name, "set" if present else "not set")


app = create_app()
