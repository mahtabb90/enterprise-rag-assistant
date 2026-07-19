"""
Configuration settings for the Enterprise RAG Assistant.

Centralizes basic application metadata and default pipeline parameters.
"""

# Application Metadata
APP_TITLE = "Enterprise RAG Assistant"
APP_DESCRIPTION = (
    "A clean, cloud-ready Retrieval-Augmented Generation (RAG) assistant "
    "designed to demonstrate enterprise document ingestion, chunking, retrieval, "
    "and answer generation."
)
VERSION = "0.1.0"

# Default RAG Parameters (for future pipeline expansion)
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 40
