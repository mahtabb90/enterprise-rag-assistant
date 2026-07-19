"""
Unit tests for keyword-based retrieval module.
"""

from app.rag.retrieve import retrieve_context


def test_retrieve_context_matching():
    """Tests that relevant chunks are retrieved based on keyword overlap."""
    sample_chunks = [
        {"source": "doc1.txt", "chunk_id": 1, "text": "Password security requires 12 characters and MFA."},
        {"source": "doc2.txt", "chunk_id": 2, "text": "Annual leave policy allows 25 days of paid vacation."},
        {"source": "doc3.txt", "chunk_id": 3, "text": "API rate limits allow 1000 requests per minute."}
    ]
    
    # Query with punctuation and mixed casing
    results = retrieve_context("What is the Password Security requirement?", sample_chunks, top_k=2)
    
    assert len(results) >= 1
    assert results[0]["chunk_id"] == 1
    assert "Password" in results[0]["text"]


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
