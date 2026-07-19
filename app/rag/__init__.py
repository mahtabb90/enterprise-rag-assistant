"""
RAG (Retrieval-Augmented Generation) Pipeline Package.

Contains modules for text chunking, document ingestion, context retrieval,
and answer generation.
"""

from app.rag.chunker import chunk_text

__all__ = ["chunk_text"]
