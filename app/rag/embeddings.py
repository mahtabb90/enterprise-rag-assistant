"""
Vertex AI Embedding Module.

Provides a clean abstraction for generating dense semantic embeddings using
the Google Gen AI SDK (google-genai >= 2.25.0) with the Vertex AI backend.

Indexing vs Query-Time Distinction
------------------------------------
This module exposes two distinct embedding functions with explicitly different
task types — a requirement for high-quality semantic retrieval:

    embed_documents()
        Task type:  RETRIEVAL_DOCUMENT
        Phase:      INDEXING — call once when building the document index.
        Purpose:    Produces embeddings optimised for stored document content.

    embed_query()
        Task type:  RETRIEVAL_QUERY
        Phase:      QUERY-TIME — call once per user question.
        Purpose:    Produces embeddings optimised for incoming search queries.

Both task types produce vectors in the same 768-dimensional space, so cosine
similarity between document and query vectors is meaningful and directly
comparable.

Authentication
--------------
Uses Application Default Credentials (ADC). No API keys or hardcoded
credentials are accepted, stored, or required.

For local development, authenticate once with:
    gcloud auth application-default login

Configuration
-------------
Reads GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION from environment
variables, which are populated from .env by config.py before this module
is used. See .env.example for the expected variable names.
"""

import os
from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONALITY = 768


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def create_vertex_client() -> genai.Client:
    """Create and return a Google Gen AI client configured for Vertex AI.

    Reads project and location from environment variables.  Uses Application
    Default Credentials (ADC) — no API key is required or used.

    Environment variables
    --------------------
    GOOGLE_CLOUD_PROJECT   Required.  Your Google Cloud project ID.
    GOOGLE_CLOUD_LOCATION  Optional.  Defaults to "global".

    Returns:
        genai.Client: A client targeting the Vertex AI backend.

    Raises:
        ValueError: If GOOGLE_CLOUD_PROJECT is not set in the environment.
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

    if not project:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT environment variable is not set. "
            "Add it to your .env file or shell environment. "
            "See .env.example for the expected format."
        )

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )


# ---------------------------------------------------------------------------
# Document embedding  (indexing phase)
# ---------------------------------------------------------------------------

def embed_documents(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """Generate dense embeddings for a list of document texts.

    INDEXING PHASE — call this once when building the document index.
    The resulting vectors should be stored and reused for all subsequent
    queries.  Do not re-embed documents on every user question.

    Uses task type RETRIEVAL_DOCUMENT, which instructs the model to produce
    embeddings optimised for representing stored document content.

    Args:
        client: A configured Vertex AI Gen AI client (from create_vertex_client()).
        texts:  List of document text strings to embed.  For very large corpora,
                consider batching to stay within API request size limits.

    Returns:
        list[list[float]]: One EMBEDDING_DIMENSIONALITY-dimensional (768) float
            vector per input text, in the same order as the input list.
    """
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBEDDING_DIMENSIONALITY,
        ),
    )
    # Convert to plain Python lists for consistent downstream handling.
    return [list(embedding.values) for embedding in response.embeddings]


# ---------------------------------------------------------------------------
# Query embedding  (query-time phase)
# ---------------------------------------------------------------------------

def embed_query(client: genai.Client, text: str) -> list[float]:
    """Generate a dense embedding for a single query string.

    QUERY-TIME PHASE — call this once per user question.

    Uses task type RETRIEVAL_QUERY, which instructs the model to produce
    embeddings optimised for representing an incoming search query.  The
    resulting vector lives in the same 768-dimensional space as
    RETRIEVAL_DOCUMENT vectors, enabling direct cosine similarity comparison.

    Args:
        client: A configured Vertex AI Gen AI client (from create_vertex_client()).
        text:   The user's query string.

    Returns:
        list[float]: A single EMBEDDING_DIMENSIONALITY-dimensional (768) float vector.
    """
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIMENSIONALITY,
        ),
    )
    return list(response.embeddings[0].values)
