"""FastAPI application factory.

Run with:  python -m app        (or: uvicorn app.main:app --reload --port 8080)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ingestion import IngestionService
from agent.rag import AgenticRAG

from .bootstrap import build_services
from .routes.graph import router as graph_router
from .routes.ingest import router as ingest_router
from .routes.rag import router as rag_router

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


def create_app(rag: AgenticRAG | None = None, ingestor: IngestionService | None = None) -> FastAPI:
    """Build the app. Pass services to inject them (tests); otherwise they are built from .env on startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if rag is None:
            services = build_services()
            app.state.rag = services.rag
            app.state.ingestor = ingestor or services.ingestor
        else:
            app.state.rag = rag
            # An injected agent never triggers a .env build: share its embedder/store/graph instead.
            app.state.ingestor = ingestor or IngestionService(rag.retriever.embedder, rag.retriever.store, graph=rag.graph_store)
        app.state.graph = app.state.rag.graph_store  # None when the graph is disabled
        yield

    app = FastAPI(title="LankaLaw Intelligence Engine", version="0.4.0", lifespan=lifespan)
    app.include_router(rag_router)
    app.include_router(ingest_router)
    app.include_router(graph_router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
