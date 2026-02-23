from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_service.api.routes import auth, health, ingest, query
from rag_service.core.config import get_settings
from rag_service.core.logging import get_logger, setup_logging
from rag_service.db.postgres import init_session_factory

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    settings = get_settings()

    # Setup logging
    setup_logging()
    logger.info("Starting RAG service", env=settings.app_env)

    # Initialize database session factory
    init_session_factory(settings)
    logger.info("Database session factory initialized")

    yield

    # Cleanup
    logger.info("Shutting down RAG service")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="RAG Service",
        description="Retrieval Augmented Generation Service for kbfinai",
        version="0.1.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")
    app.include_router(ingest.router, prefix="/api/v1")

    return app


app = create_app()


def run() -> None:
    """Run the application with uvicorn."""
    settings = get_settings()
    uvicorn.run(
        "rag_service.main:app",
        host=settings.host,
        port=settings.port,
        reload=not settings.is_production,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
