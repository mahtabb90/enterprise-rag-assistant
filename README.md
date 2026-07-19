# Enterprise RAG Assistant

A modern Python 3.13 demonstration application designed to showcase an enterprise-grade Retrieval-Augmented Generation (RAG) architecture built with a clean local foundation and cloud-ready modular components.

---

## 🎯 Overview

The **Enterprise RAG Assistant** provides an interactive web workspace built with Streamlit. It demonstrates how organizations can ingest internal documents, break text into semantic chunks, generate vector embeddings, retrieve contextually relevant information, and leverage Large Language Models (LLMs) to answer queries with high accuracy.

---

## 🚀 Progress & Milestones

- **Day 1: Project Foundation & Layout**
  - Modular project structure (`app/` package, `rag/` submodules, `tests/`).
  - Streamlit UI entry point and unit test suite setup.
- **Day 2: Local RAG Pipeline Prototype**
  - **Local Ingestion**: Scans and parses `.txt` and `.md` enterprise documents from `data/sample_docs/`.
  - **Document Chunking**: Structured sliding-window text chunking with metadata tracking (`source`, `chunk_id`).
  - **Normalized Keyword Retrieval**: Token-based keyword overlap matching with lowercasing and punctuation stripping.
  - **Grounded Answer Generation**: Local response synthesis presenting retrieved context snippets and source citations.

---

## 🛠️ Planned Learning Stack

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
│       ├── ingest.py       # Local document ingestion module
│       ├── retrieve.py     # Normalized keyword retrieval module
│       └── generate.py     # Local grounded answer generation module
├── data/
│   └── sample_docs/        # Sample enterprise documents (security, HR, FAQ)
├── tests/
│   ├── test_chunker.py     # Unit tests for text chunker
│   ├── test_ingest.py      # Unit tests for document ingestion
│   └── test_retrieve.py    # Unit tests for keyword retrieval
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
