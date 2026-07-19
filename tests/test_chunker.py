"""
Unit tests for the text chunking module.
"""

import pytest
from app.rag.chunker import chunk_text, chunk_documents


def test_chunk_text_empty_input():
    """Tests that empty or whitespace-only input returns an empty list."""
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_short_input():
    """Tests that text shorter than chunk_size returns a single chunk."""
    text = "Short sample text."
    result = chunk_text(text, chunk_size=50, overlap=10)
    assert len(result) == 1
    assert result[0] == text


def test_chunk_text_splitting():
    """Tests that long text is correctly split into overlapping chunks."""
    text = "12345678901234567890"  # 20 characters long
    chunk_size = 10
    overlap = 2
    
    result = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    
    assert len(result) == 3
    assert result[0] == "1234567890"
    assert result[1] == "9012345678"
    assert result[2] == "7890"


def test_chunk_text_invalid_overlap():
    """Tests that overlap greater than or equal to chunk_size raises ValueError."""
    with pytest.raises(ValueError):
        chunk_text("Sample text", chunk_size=10, overlap=10)


def test_chunk_documents():
    """Tests chunking a list of document objects with metadata preservation."""
    sample_docs = [
        {"source": "doc1.txt", "text": "Hello world from document one."},
        {"source": "doc2.txt", "text": "Second document content for chunking."}
    ]
    
    chunks = chunk_documents(sample_docs, chunk_size=15, overlap=5)
    
    assert len(chunks) > 0
    # Check that metadata fields exist on each chunk dictionary
    for idx, chunk in enumerate(chunks, start=1):
        assert "source" in chunk
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert chunk["chunk_id"] == idx

    # Verify first chunk source
    assert chunks[0]["source"] == "doc1.txt"
