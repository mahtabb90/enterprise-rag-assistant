"""
Unit tests for keyword-based retrieval module.
"""

from app.rag.ingest import ingest_documents
from app.rag.chunker import chunk_documents
from app.rag.retrieve import retrieve_context


def test_retrieve_context_matching():
    """Tests that relevant chunks are retrieved based on keyword overlap."""
    sample_chunks = [
        {"source": "security_guidelines.txt", "chunk_id": 1, "text": "Password security requires 12 characters and MFA."},
        {"source": "hr_policy.txt", "chunk_id": 2, "text": "Annual leave policy allows 25 days of paid vacation."},
        {"source": "product_faq.txt", "chunk_id": 3, "text": "API rate limits allow 1000 requests per minute."}
    ]
    
    results = retrieve_context("What is the Password Security requirement?", sample_chunks, top_k=2)
    
    assert len(results) >= 1
    assert results[0]["chunk_id"] == 1
    assert "security_guidelines.txt" in results[0]["source"]


def test_retrieve_context_password_query():
    """Tests that password requirements query retrieves security guidelines."""
    docs = ingest_documents("data/sample_docs")
    chunks = chunk_documents(docs)
    
    results = retrieve_context("What are the password requirements?", chunks)
    assert len(results) >= 1
    assert results[0]["source"] == "security_guidelines.txt"


def test_retrieve_context_annual_leave_query():
    """Tests that annual leave query retrieves HR policy."""
    docs = ingest_documents("data/sample_docs")
    chunks = chunk_documents(docs)
    
    results = retrieve_context("How many days of annual leave do I get?", chunks)
    assert len(results) >= 1
    assert results[0]["source"] == "hr_policy.txt"


def test_retrieve_context_api_rate_limits_query():
    """Tests that API rate limits query retrieves product FAQ."""
    docs = ingest_documents("data/sample_docs")
    chunks = chunk_documents(docs)
    
    results = retrieve_context("What are the API rate limits?", chunks)
    assert len(results) >= 1
    assert results[0]["source"] == "product_faq.txt"


def test_retrieve_context_company_revenue_query():
    """Tests that company revenue query returns an empty list since information is absent."""
    docs = ingest_documents("data/sample_docs")
    chunks = chunk_documents(docs)
    
    results = retrieve_context("What is the company revenue?", chunks)
    assert results == []


def test_retrieve_context_no_match():
    """Tests that queries with no keyword overlap return an empty list."""
    sample_chunks = [
        {"source": "doc1.txt", "chunk_id": 1, "text": "Password security policy details."},
        {"source": "doc2.txt", "chunk_id": 2, "text": "Annual leave guidelines."}
    ]
    
    results = retrieve_context("quantum computing astrophysics", sample_chunks, top_k=3)
    assert results == []


def test_retrieve_context_empty_query():
    """Tests that empty or whitespace-only queries return an empty list."""
    sample_chunks = [{"source": "doc1.txt", "chunk_id": 1, "text": "Sample content."}]
    assert retrieve_context("", sample_chunks) == []
    assert retrieve_context("   !!! ", sample_chunks) == []
