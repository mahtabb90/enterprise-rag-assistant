"""
Unit tests for the text chunking module.
"""

import pytest
from app.rag.chunker import chunk_text


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
    
    # step = 10 - 2 = 8
    # Chunk 0: start 0, end 10 -> "1234567890"
    # Chunk 1: start 8, end 18 -> "9012345678"
    # Chunk 2: start 16, end 26 -> "7890"
    result = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    
    assert len(result) == 3
    assert result[0] == "1234567890"
    assert result[1] == "9012345678"
    assert result[2] == "7890"


def test_chunk_text_invalid_overlap():
    """Tests that overlap greater than or equal to chunk_size raises ValueError."""
    with pytest.raises(ValueError):
        chunk_text("Sample text", chunk_size=10, overlap=10)
