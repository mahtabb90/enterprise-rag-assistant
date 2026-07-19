"""
Unit tests for the document ingestion module.
"""

from app.rag.ingest import ingest_documents


def test_ingest_documents_sample_docs():
    """Tests loading documents from the sample_docs directory."""
    docs = ingest_documents("data/sample_docs")
    
    assert len(docs) == 3
    sources = {doc["source"] for doc in docs}
    assert "security_guidelines.txt" in sources
    assert "hr_policy.txt" in sources
    assert "product_faq.txt" in sources
    
    for doc in docs:
        assert isinstance(doc["text"], str)
        assert len(doc["text"]) > 0


def test_ingest_documents_missing_folder():
    """Tests that a non-existent directory returns an empty list gracefully."""
    docs = ingest_documents("data/non_existent_folder_path")
    assert docs == []
