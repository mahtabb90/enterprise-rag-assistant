"""
RAG (Retrieval-Augmented Generation) Pipeline Package.

Contains modules for document ingestion, text chunking, TF-IDF vector retrieval,
keyword retrieval, and local answer generation.
"""

from app.rag.ingest import ingest_documents
from app.rag.chunker import chunk_text, chunk_documents
from app.rag.vectorization import build_vectorizer, vectorize_corpus, vectorize_query
from app.rag.retrieve import retrieve_context, prepare_tfidf_index, retrieve_tfidf
from app.rag.generate import generate_answer

__all__ = [
    "ingest_documents",
    "chunk_text",
    "chunk_documents",
    "build_vectorizer",
    "vectorize_corpus",
    "vectorize_query",
    "retrieve_context",
    "prepare_tfidf_index",
    "retrieve_tfidf",
    "generate_answer",
]
