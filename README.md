# Enterprise RAG Assistant

A modern Python 3.13 demonstration application showcasing an enterprise-grade
Retrieval-Augmented Generation (RAG) architecture with a clean local foundation
and Vertex AI semantic embeddings.

---

## Overview

The **Enterprise RAG Assistant** provides an interactive web workspace built with
Streamlit. It demonstrates how organizations can ingest internal documents, split
text into semantic chunks, generate dense vector embeddings, retrieve contextually
relevant information, and synthesize grounded answers.

---

## Implemented Capabilities

### Local RAG Foundation

- Modular project structure (`app/` package, `rag/` submodules, `tests/`).
- Streamlit UI entry point and automated unit test suite.

### Local RAG Pipeline

- **Local Ingestion**: Scans and parses `.txt` and `.md` enterprise documents from `data/sample_docs/`.
- **Document Chunking**: Structured sliding-window text chunking with metadata tracking (`source`, `chunk_id`).
- **Grounded Answer Generation**: Local response synthesis presenting retrieved context snippets and source citations.

### TF-IDF Vector Retrieval

- **TF-IDF Lexical Vectorization** (`vectorization.py`): Builds a `TfidfVectorizer` fitted on the document corpus.
- **Indexing / Retrieval Separation**: `prepare_tfidf_index()` fits the vectorizer and transforms the corpus **once** at startup. `retrieve_tfidf()` transforms only the incoming query at runtime.
- **Cosine Similarity Ranking**: Query and chunk vectors are compared using cosine similarity (0.0-1.0).
- **Keyword Fallback**: If TF-IDF finds no result above its threshold, keyword overlap matching is used as a fallback.
- **Consistent Result Schema**: All retrieval paths return `source`, `chunk_id`, `text`, `score`, and `retrieval_method`.

### Vertex AI Semantic Embeddings (Primary Retrieval)

- **Model**: `gemini-embedding-001` via the `google-genai` SDK (`>=2.25.0,<3.0`).
- **Output dimensionality**: 768-dimensional dense floating-point vectors.
- **Task types**:
  - `RETRIEVAL_DOCUMENT` -- used during the **indexing phase** to embed stored document chunks.
  - `RETRIEVAL_QUERY` -- used at **query time** to embed the incoming user question.
  - These are distinct task types: the embedding model optimises the vector differently depending on whether it represents a document or a query. Both task types produce vectors in the same 768-dimensional space so cosine similarity is meaningful.
- **Indexing phase** (`prepare_vertex_index`): Document chunks are embedded **once** at application startup and stored in a cached index. Embeddings are not regenerated for every query -- only the incoming query is embedded at runtime.
- **Query-time phase** (`retrieve_vertex`): The query is embedded with `RETRIEVAL_QUERY`, cosine similarity is computed against all cached document embeddings, and the top-ranked chunks above the configured threshold are returned.
- **Authentication**: Application Default Credentials (ADC). No API keys. No credentials stored in the repository. See *Local Development Setup* below.

### Three-Tier Fallback Architecture

Retrieval occurs in the following order:

```
Vertex AI semantic retrieval       (primary)
  |  results above VERTEX_MIN_SCORE --> return them
  |  [] (all chunks below threshold) OR exception --> fall through
  v
TF-IDF vector retrieval            (secondary)
  |  if TF-IDF finds no result above its threshold
  v
Keyword overlap retrieval          (tertiary)
  |  if keyword retrieval finds no meaningful match
  v
[]  -->  no-answer response shown; context expander not displayed
```

- Vertex errors are **not** silently swallowed. A warning is logged so the problem is visible during local development.
- The application remains fully usable when Vertex credentials or cloud access are unavailable. `VERTEX_ENABLED` is `False` when `GOOGLE_CLOUD_PROJECT` is unset, and the pipeline falls through to TF-IDF automatically.
- If Vertex returns `[]` because no chunk exceeds `VERTEX_MIN_SCORE`, the pipeline continues to TF-IDF and keyword retrieval. Rejected low-confidence chunks are never shown in the UI. See *VERTEX_MIN_SCORE Calibration* for observed scores that informed the default threshold.

### Semantic vs Lexical Retrieval

TF-IDF is a **lexical** method: it scores chunks by exact word overlap. It cannot match synonyms or paraphrases.

Vertex AI embeddings are **semantic**: the model captures meaning regardless of exact wording. For example:

| Query | Document text | TF-IDF | Vertex AI |
|---|---|---|---|
| "What is the holiday allowance for workers?" | "Full-time employees receive 25 days of paid annual leave per calendar year." | `[]` (no shared keywords) | Retrieves the chunk (semantic match) |
| "What are the password requirements?" | "All passwords must be at least 12 characters..." | Retrieves the chunk (lexical match) | Also retrieves the chunk |

The `test_semantic_vs_lexical_contrast` test in `tests/test_vertex_retrieve.py` demonstrates this architectural difference using controlled mock embeddings.

### Cost-Conscious Design

- Document embeddings are computed **once per Streamlit session** via `@st.cache_resource`, keyed on the document chunks. Subsequent queries only embed the incoming question.
- No Cloud Run, BigQuery, Cloud Storage, vector databases, or other paid infrastructure is provisioned. Only the Vertex AI Embeddings API is used.
- `VERTEX_MIN_SCORE` filtering is performed locally after the single query embedding call; it has no effect on API call volume.

---

## Repository Structure

```text
enterprise-rag-assistant/
+-- app/
|   +-- __init__.py             # Application package initializer
|   +-- main.py                 # Streamlit application entry point
|   +-- config.py               # App settings + cloud configuration (env-variable-driven)
|   +-- rag/
|       +-- __init__.py         # RAG pipeline package initializer
|       +-- chunker.py          # Text chunking logic with overlap support
|       +-- embeddings.py       # Vertex AI embedding client and functions
|       +-- generate.py         # Local grounded answer generation module
|       +-- ingest.py           # Local document ingestion module
|       +-- retrieve.py         # Three-tier retrieval: Vertex / TF-IDF / keyword
|       +-- vectorization.py    # TF-IDF lexical vectorization primitives
+-- data/
|   +-- sample_docs/            # Sample enterprise documents (security, HR, FAQ)
+-- tests/
|   +-- test_chunker.py         # Unit tests for text chunker
|   +-- test_generate.py        # Unit tests for answer generation
|   +-- test_ingest.py          # Unit tests for document ingestion
|   +-- test_retrieve.py        # Unit tests for keyword retrieval
|   +-- test_tfidf_retrieve.py  # Unit tests for TF-IDF vector retrieval
|   +-- test_vertex_retrieve.py # Unit tests for Vertex AI semantic retrieval (mocked)
+-- .env.example                # Public-safe placeholder env file (safe to commit)
+-- .gitignore                  # Ignored files (virtualenvs, .env, secrets, caches)
+-- pytest.ini                  # Pytest configuration
+-- README.md                   # Project documentation
+-- requirements.txt            # Project dependencies
```

---

## Quickstart Guide

### 1. Environment Setup

Clone the repository and create a Python 3.13 virtual environment:

```bash
python -m venv .venv
```

Activate the environment:
- **Windows (PowerShell)**: `.venv\Scripts\Activate.ps1`
- **macOS / Linux**: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your own values:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=true
VERTEX_MIN_SCORE=0.6
```

> **Important**: `.env` is git-ignored and must never be committed. Only `.env.example` (with placeholder values) belongs in the repository.

### 3. Local Development Authentication (Vertex AI)

Authenticate with Application Default Credentials (ADC) once:

```bash
gcloud auth application-default login
```

No API keys are used. No credentials are stored in the codebase. If `GOOGLE_CLOUD_PROJECT` is not set, the application runs in TF-IDF-only mode automatically.

### 4. Run the Streamlit Application

```bash
streamlit run app/main.py
```

Open `http://localhost:8501` in your browser.

### 5. Run Unit Tests

```bash
pytest
```

All Vertex AI tests use mocked embeddings -- no live API calls are made during testing.

---

## VERTEX_MIN_SCORE Calibration

The `VERTEX_MIN_SCORE` setting controls the minimum cosine similarity a Vertex AI
embedding result must reach to be accepted. Chunks that score below this value are
**rejected** — they are not returned to the caller and do not appear in the UI.

### Default: 0.6

The default of `0.6` was calibrated from **live Vertex AI testing** against the
fictional sample corpus in this repository (`gemini-embedding-001`, `global` endpoint).

| Query | Type | Top score | Result at ≥ 0.6 |
|---|---|---|---|
| "What are the password requirements?" | Direct match | 0.7105 | ✅ Accepted |
| "How many days of annual leave do I get?" | Direct match | 0.6947 | ✅ Accepted |
| "How much vacation time do employees get?" | Semantic synonym | 0.7275 | ✅ Accepted |
| "Who was Julius Caesar?" | Completely unrelated | 0.5092 | ❌ Rejected |

The threshold sits in the gap between the lowest relevant score (0.6947) and the
highest unrelated score (0.5092), providing a clean separation for this corpus.

### Recalibration required when

This value is **not a universal threshold**. Recalibrate `VERTEX_MIN_SCORE`
whenever any of the following change:

- The document corpus or its content
- The chunking strategy or chunk size parameters
- The embedding model or model version
- Observed retrieval behaviour in a new environment

### Calibration procedure

1. Run several representative queries in the Streamlit UI.
2. Inspect per-chunk similarity scores in the *"View Retrieved Sources & Context"* expander.
3. Collect scores across three query categories:
   - **Clearly relevant** — expect high scores (e.g. > 0.7)
   - **Semantic / synonym** — expect moderate scores (e.g. 0.6 – 0.75)
   - **Completely unrelated** — expect low scores (e.g. < 0.55)
4. Set `VERTEX_MIN_SCORE` to a value in the gap between your *unrelated ceiling*
   and your *semantic floor*.
5. Update `VERTEX_MIN_SCORE` in your `.env` file to override the default.

---

## Security and Credentials

- No credentials, project IDs, API keys, billing account IDs, or access tokens are stored anywhere in this repository.
- The `.env` file is git-ignored. Only `.env.example` (placeholder values) is committed.
- Authentication uses Application Default Credentials (ADC) exclusively.
- The Streamlit UI does not display any cloud configuration, project identifiers, or credential information.

---

## Planned Architecture

- **Containerization**: Docker
- **Cloud Infrastructure**: Cloud Run, Cloud Storage, BigQuery
- **DevOps & IaC**: GitHub Actions, Terraform

---

## License

Distributed under the MIT License. See `LICENSE` for details.
