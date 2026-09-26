"""
Configuration settings for the Enterprise RAG Assistant.

Centralizes application metadata and default pipeline parameters.
Cloud configuration is read from environment variables so no credentials
or project identifiers are hardcoded in this file or anywhere in the
repository.

For local development, copy .env.example to .env and fill in your own
values. The .env file is git-ignored and must never be committed.
"""

import os
from dotenv import load_dotenv

# Load .env for local development. This is a no-op when .env is absent
# (e.g. in CI environments where variables are injected directly).
load_dotenv()

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------

APP_TITLE = "Enterprise RAG Assistant"
APP_DESCRIPTION = (
    "A clean, cloud-ready Retrieval-Augmented Generation (RAG) assistant "
    "designed to demonstrate enterprise document ingestion, chunking, retrieval, "
    "and answer generation."
)
VERSION = "0.2.0"

# ---------------------------------------------------------------------------
# Default RAG pipeline parameters
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 40

# ---------------------------------------------------------------------------
# Google Cloud / Vertex AI configuration
# ---------------------------------------------------------------------------
# All values are read from environment variables so no credentials or project
# identifiers are stored in the codebase.  See .env.example for variable names.

GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
GOOGLE_GENAI_USE_VERTEXAI = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "true")

# Convenience flag. True only when a Cloud project is configured.
# Used as a single gate across the application to enable/disable Vertex features.
VERTEX_ENABLED = bool(GOOGLE_CLOUD_PROJECT)

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONALITY = 768

# ---------------------------------------------------------------------------
# Vertex AI semantic relevance threshold
# ---------------------------------------------------------------------------
# Cosine similarity cut-off for Vertex retrieval results (range 0.0 - 1.0).
#
# CALIBRATION NOTICE
# ------------------
# The default value of 0.6 was calibrated from live Vertex AI testing against
# the FICTIONAL SAMPLE CORPUS included in this repository.  It is NOT a
# universal threshold and must be recalibrated whenever any of the following
# change:
#   - the document corpus or its content
#   - the chunking strategy or chunk size
#   - the embedding model or model version
#   - observed retrieval behaviour in production
#
# Observed scores on the fictional sample corpus (gemini-embedding-001, global):
#   Relevant password query                         : 0.7105  (accepted at >= 0.6)
#   Relevant annual leave query                     : 0.6947  (accepted at >= 0.6)
#   Semantic synonym "vacation time / annual leave" : 0.7275  (accepted at >= 0.6)
#   Unrelated query "Who was Julius Caesar?"        : 0.5092  (REJECTED at >= 0.6)
#
# The 0.6 threshold sits between the lowest relevant score (0.6947) and the
# highest observed unrelated score (0.5092), providing a clean separation for
# this corpus.  A different corpus will produce different score distributions.
#
# Calibration procedure:
#   Run representative queries and inspect per-chunk scores in the
#   "View Retrieved Sources & Context" expander:
#     1. Clearly relevant queries   --> expect high scores (e.g. > 0.7)
#     2. Semantic / synonym queries --> expect moderate scores (e.g. 0.6 - 0.75)
#     3. Completely unrelated       --> expect low scores  (e.g. < 0.55)
#   Set VERTEX_MIN_SCORE in .env to a value between the unrelated ceiling
#   and the semantic floor for your specific corpus.
#
# Override without code changes by setting VERTEX_MIN_SCORE in .env.
VERTEX_MIN_SCORE = float(os.environ.get("VERTEX_MIN_SCORE", "0.6"))

