"""Ingestion: loaders -> chunker -> embed -> vector store, plus document management.

Kept outside `agent/` on purpose. The agent only reads the knowledge base; this package
is the only code that writes to it. It depends on the agent through small interfaces
(`Embedder`, `VectorStore`, `Document`), never the other way round.
"""
from .chunker import chunk_text
from .config import IngestSettings
from .graph_extractor import ExtractedFacts, GraphExtractor, LLMGraphExtractor
from .loaders import EmptyDocumentError, UnsupportedFileError, extract_text, supported_extensions
from .schemas import IndexedDocument, IngestResult
from .service import FileTooLargeError, IngestionService, doc_id_for

__all__ = [
    "EmptyDocumentError",
    "ExtractedFacts",
    "FileTooLargeError",
    "GraphExtractor",
    "LLMGraphExtractor",
    "IndexedDocument",
    "IngestResult",
    "IngestSettings",
    "IngestionService",
    "UnsupportedFileError",
    "chunk_text",
    "doc_id_for",
    "extract_text",
    "supported_extensions",
]
