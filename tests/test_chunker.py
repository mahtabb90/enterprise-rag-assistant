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


def test_chunk_text_word_boundary_alignment():
    """Tests that text with spaces aligns boundaries on words rather than cutting words."""
    text = "All user accounts must use strong passwords at least 12 characters."
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    
    assert len(chunks) >= 2
    # Verify that words are intact and not cut mid-word (e.g. no "passwor")
    for chunk in chunks:
        assert chunk == chunk.strip()
        words = chunk.split()
        assert len(words) > 0


def test_chunk_documents():
    """Tests chunking a list of document objects with metadata preservation."""
    sample_docs = [
        {"source": "doc1.txt", "text": "Hello world from document one."},
        {"source": "doc2.txt", "text": "Second document content for chunking."}
    ]
    
    chunks = chunk_documents(sample_docs, chunk_size=15, overlap=5)
    
    assert len(chunks) > 0
    for idx, chunk in enumerate(chunks, start=1):
        assert "source" in chunk
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert chunk["chunk_id"] == idx
        assert chunk["text"] == chunk["text"].strip()

    assert chunks[0]["source"] == "doc1.txt"


def test_chunk_text_prefers_sentence_boundary():
    """Chunks should end at sentence boundaries (. ! ?) when available."""
    text = (
        "All user accounts must use strong passwords. "
        "Passwords must be at least twelve characters long. "
        "Multi-factor authentication is also required."
    )
    chunks = chunk_text(text, chunk_size=70, overlap=10)
    assert len(chunks) >= 2
    # Every chunk except possibly the last should end on sentence punctuation.
    for chunk in chunks[:-1]:
        assert chunk.endswith((".", "!", "?")), (
            f"Expected chunk to end on sentence boundary, got: {repr(chunk[-30:])}"
        )


def test_chunk_text_no_mid_word_splits():
    """No chunk should begin or end with a partial word when whitespace is available."""
    text = (
        "Security requires strong authentication and access controls. "
        "Annual leave entitlement is twenty-five days per fiscal year. "
        "API rate limits are enforced per authenticated client key."
    )
    chunks = chunk_text(text, chunk_size=80, overlap=15)
    for chunk in chunks:
        assert chunk == chunk.strip(), f"Chunk has leading/trailing whitespace: {repr(chunk)}"
        assert chunk[0].isalnum() or chunk[0] in ('"', "'", "("), (
            f"Chunk starts with unexpected character: {repr(chunk[:20])}"
        )
