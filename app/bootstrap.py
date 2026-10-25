"""Composition root: builds the agent (agentic RAG) and the ingestion service from .env,
sharing one embedder, one vector store and one knowledge graph, so what is ingested is
immediately searchable.

The API layer calls `build_services()` at startup and nothing else constructs objects.
"""
from __future__ import annotations

from dataclasses import dataclass

from ingestion import IngestionService, IngestSettings, LLMGraphExtractor
from agent.rag import AgenticRAG, Embedder, GeminiEmbedder, RAGSettings
from agent.services.db.graphstore import GraphStore, KuzuGraphStore
from agent.services.db.vectorstore import ChromaVectorStore, VectorStore
from agent.services.llm import FallbackLLM, LLMSettings


@dataclass
class Services:
    rag: AgenticRAG
    ingestor: IngestionService
    embedder: Embedder
    store: VectorStore
    graph: GraphStore | None


def build_services(dotenv_path: str | None = None) -> Services:
    llm_settings = LLMSettings.from_env(dotenv_path)
    rag_settings = RAGSettings.from_env(dotenv_path)
    ingest = IngestSettings.from_env(dotenv_path)
    if not llm_settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is required for embeddings")

    embedder = GeminiEmbedder(llm_settings.gemini_api_key, rag_settings.embedding_model, rag_settings.embedding_dim)
    store = ChromaVectorStore(rag_settings.chroma_path, rag_settings.collection)
    graph: GraphStore | None = KuzuGraphStore(rag_settings.graph_path) if rag_settings.graph_enabled else None
    llm = FallbackLLM.from_settings(llm_settings)
    rag = AgenticRAG.build(llm_settings, rag_settings, embedder, store, graph, llm=llm)
    ingestor = IngestionService(
        embedder, store, ingest, graph=graph, extractor=LLMGraphExtractor(llm) if graph is not None else None
    )
    return Services(rag=rag, ingestor=ingestor, embedder=embedder, store=store, graph=graph)
