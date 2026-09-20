# Enterprise RAG Assistant

A modern Python 3.13 demonstration application designed to showcase an enterprise-grade Retrieval-Augmented Generation (RAG) architecture built with a clean local foundation and cloud-ready modular components.

---

## 🎯 Overview

The **Enterprise RAG Assistant** provides an interactive web workspace built with Streamlit. It demonstrates how organizations can ingest internal documents, break text into semantic chunks, generate vector embeddings, retrieve contextually relevant information, and leverage Large Language Models (LLMs) to answer queries with high accuracy.

---

## ✅ Implemented Capabilities

### Local RAG Foundation
- Modular project structure (`app/` package, `rag/` submodules, `tests/`).
- Streamlit UI entry point and automated unit test suite.

### Local RAG Pipeline
- **Local Ingestion**: Scans and parses `.txt` and `.md` enterprise documents from `data/sample_docs/`.
- **Document Chunking**: Structured sliding-window text chunking with metadata tracking (`source`, `chunk_id`).
- **Normalized Keyword Retrieval**: Token-based keyword overlap matching with lowercasing and punctuation stripping.
- **Grounded Answer Generation**: Local response synthesis presenting retrieved context snippets and source citations.

### TF-IDF Vector Retrieval
- **TF-IDF Lexical Vectorization** (`vectorization.py`): Builds a `TfidfVectorizer` fitted on the document corpus. Documents and queries are both transformed using the same fitted vectorizer, ensuring a shared feature space — a prerequisite for cosine similarity to be meaningful.
- **Indexing / Retrieval Separation**: `prepare_tfidf_index()` fits the vectorizer and transforms the corpus **once** at startup. `retrieve_tfidf()` transforms only the incoming query at runtime and compares it against the pre-built index. This mirrors the RAG pipeline distinction between offline document indexing and online query serving.
- **Cosine Similarity Ranking**: Query and chunk vectors are compared using cosine similarity (0.0–1.0). Results below the relevance threshold are excluded. Results are ranked by score descending.
- **Keyword Fallback**: If TF-IDF finds no result above its threshold, the existing keyword retrieval is used as a fallback. If keyword retrieval also finds no meaningful match, an empty list is returned and the answer-not-found message is shown.
- **Consistent Result Schema**: Both retrieval paths return `source`, `chunk_id`, `text`, `score`, and `retrieval_method` (`"tfidf"` or `"keyword"`), displayed in the Streamlit context expander.
- **Terminology note**: TF-IDF produces *lexical vectors*, not semantic embeddings. `embeddings.py` is intentionally reserved for a future stage when a real embedding model (e.g. Vertex AI `text-embedding-004`) is introduced.

---

## 🛠️ Planned Architecture

This project is built to showcase a comprehensive, end-to-end cloud and AI engineering stack:

- **Core Language & Framework**: Python 3.13, Streamlit
- **AI & RAG Architecture**: Embeddings, Vector Search, LLM Synthesis, RAG Pipeline
- **Containerization & Orchestration**: Docker
- **Cloud Infrastructure**: Google Cloud Platform (GCP), Cloud Run, Cloud Storage, BigQuery, Artifact Registry
- **DevOps & Infrastructure as Code**: GitHub Actions, Terraform

---

## 📁 Repository Structure

```text
enterprise-rag-assistant/
├── app/
│   ├── __init__.py         # Application package initializer
│   ├── main.py             # Streamlit application entry point
│   ├── config.py           # Application settings and parameters
│   └── rag/
│       ├── __init__.py     # RAG pipeline package initializer
│       ├── chunker.py      # Text chunking logic with overlap support
│       ├── generate.py     # Local grounded answer generation module
│       ├── ingest.py       # Local document ingestion module
│       ├── retrieve.py     # TF-IDF vector retrieval + keyword fallback
│       └── vectorization.py# TF-IDF lexical vectorization primitives
├── data/
│   └── sample_docs/        # Sample enterprise documents (security, HR, FAQ)
├── tests/
│   ├── test_chunker.py     # Unit tests for text chunker
│   ├── test_generate.py    # Unit tests for answer generation
│   ├── test_ingest.py      # Unit tests for document ingestion
│   ├── test_retrieve.py    # Unit tests for keyword retrieval
│   └── test_tfidf_retrieve.py  # Unit tests for TF-IDF vector retrieval
├── .gitignore              # Ignored files (virtualenvs, secrets, caches)
├── pytest.ini              # Pytest configuration
├── README.md               # Project documentation
└── requirements.txt        # Project dependencies
```

---

## 🚀 Quickstart Guide

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

### 2. Run the Streamlit Application

Launch the local interactive UI:

```bash
streamlit run app/main.py
```

Open `http://localhost:8501` in your browser to view the application.

### 3. Run Unit Tests

Execute the automated test suite using pytest:

```bash
pytest
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
