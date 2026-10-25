"""FastAPI dependencies. Routes never construct services themselves."""
from __future__ import annotations

from fastapi import HTTPException, Request

from ingestion import IngestionService
from agent.rag import AgenticRAG
from agent.services.db.graphstore import GraphStore


def get_rag(request: Request) -> AgenticRAG:
    return request.app.state.rag


def get_ingestor(request: Request) -> IngestionService:
    return request.app.state.ingestor


def get_graph(request: Request) -> GraphStore:
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(status_code=404, detail="knowledge graph is disabled (GRAPH_ENABLED=false)")
    return graph
