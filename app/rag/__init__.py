"""
RAG (Retrieval-Augmented Generation) Pipeline Package.

Contains modules for document ingestion, text chunking, keyword retrieval,
and local answer generation.
"""

from app.rag.ingest import ingest_documents
from app.rag.chunker import chunk_text, chunk_documents
from app.rag.retrieve import retrieve_context
from app.rag.generate import generate_answer

__all__ = [
    "ingest_documents",
    "chunk_text",
    "chunk_documents",
    "retrieve_context",
    "generate_answer",
]
