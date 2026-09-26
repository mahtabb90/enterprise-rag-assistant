"""
RAG (Retrieval-Augmented Generation) Pipeline Package.

Contains modules for document ingestion, text chunking, TF-IDF vector
retrieval, keyword retrieval, Vertex AI semantic retrieval, and local
answer generation.
"""

from app.rag.ingest import ingest_documents
from app.rag.chunker import chunk_text, chunk_documents
from app.rag.vectorization import build_vectorizer, vectorize_corpus, vectorize_query
from app.rag.retrieve import (
    retrieve_context,
    prepare_tfidf_index,
    retrieve_tfidf,
    prepare_vertex_index,
    retrieve_vertex,
)
from app.rag.embeddings import create_vertex_client, embed_documents, embed_query
from app.rag.generate import generate_answer

__all__ = [
    # Ingestion and chunking
    "ingest_documents",
    "chunk_text",
    "chunk_documents",
    # TF-IDF vectorization primitives
    "build_vectorizer",
    "vectorize_corpus",
    "vectorize_query",
    # Retrieval -- keyword (fallback)
    "retrieve_context",
    # Retrieval -- TF-IDF (secondary)
    "prepare_tfidf_index",
    "retrieve_tfidf",
    # Retrieval -- Vertex AI semantic (primary)
    "prepare_vertex_index",
    "retrieve_vertex",
    # Vertex AI embedding client and functions
    "create_vertex_client",
    "embed_documents",
    "embed_query",
    # Answer generation
    "generate_answer",
]
