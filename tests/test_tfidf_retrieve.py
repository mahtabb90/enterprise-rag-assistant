"""
Unit tests for TF-IDF vector retrieval module.

All test queries use direct lexical overlap with the sample documents.
Synonym pairs (e.g. vacation/annual leave, quota/rate limit) are intentionally
NOT tested here -- those are reserved for a future semantic embedding stage
where TF-IDF and true semantic retrieval can be compared side by side.

Test structure:
  - Each test calls prepare_tfidf_index() once in setup to build the index.
  - retrieve_tfidf() is then called with that index -- mirroring the intended
    indexing/retrieval separation.
"""

import pytest
from app.rag.ingest import ingest_documents
from app.rag.chunker import chunk_documents
from app.rag.retrieve import prepare_tfidf_index, retrieve_tfidf


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_chunks():
    """Load and chunk the actual sample documents once for the test module."""
    docs = ingest_documents("data/sample_docs")
    return chunk_documents(docs)


@pytest.fixture(scope="module")
def real_index(real_chunks):
    """Build the TF-IDF index once for the test module.

    This mirrors the intended indexing phase: prepare once, reuse for every query.
    """
    return prepare_tfidf_index(real_chunks)


# ---------------------------------------------------------------------------
# Retrieval correctness tests (direct lexical overlap queries)
# ---------------------------------------------------------------------------

def test_tfidf_annual_leave_query_retrieves_hr_policy(real_index):
    """Query with direct lexical overlap retrieves HR policy as the top result."""
    results = retrieve_tfidf("How many days of annual leave do I get?", real_index, top_k=3)
    assert len(results) >= 1, "Expected at least one result for annual leave query"
    assert results[0]["source"] == "hr_policy.txt", (
        f"Expected hr_policy.txt as top result, got {results[0]['source']}"
    )


def test_tfidf_password_query_retrieves_security_doc():
    """Query with direct lexical overlap retrieves security guidelines as the top result.

    Uses a controlled corpus so the test is independent of real document chunk
    boundaries, which shift as the chunker improves.
    """
    chunks = [
        {
            "source": "hr_policy.txt",
            "chunk_id": 1,
            "text": "Full-time employees receive 25 days of paid annual leave per calendar year.",
        },
        {
            "source": "security_guidelines.txt",
            "chunk_id": 2,
            "text": (
                "All user accounts must use strong passwords. "
                "Password requirements include at least 12 characters, "
                "uppercase letters, lowercase letters, numbers, and special symbols."
            ),
        },
        {
            "source": "product_faq.txt",
            "chunk_id": 3,
            "text": "The API supports up to 1,000 requests per minute per authenticated key.",
        },
    ]
    index = prepare_tfidf_index(chunks)
    results = retrieve_tfidf("What are the password requirements?", index, top_k=3)
    assert len(results) >= 1, "Expected at least one result for password query"
    assert results[0]["source"] == "security_guidelines.txt", (
        f"Expected security_guidelines.txt as top result, got {results[0]['source']}"
    )


def test_tfidf_api_rate_limit_query_retrieves_faq(real_index):
    """Query with direct lexical overlap retrieves product FAQ as the top result."""
    results = retrieve_tfidf("What are the API rate limits?", real_index, top_k=3)
    assert len(results) >= 1, "Expected at least one result for API rate limits query"
    assert results[0]["source"] == "product_faq.txt", (
        f"Expected product_faq.txt as top result, got {results[0]['source']}"
    )


# ---------------------------------------------------------------------------
# Unrelated query test
# ---------------------------------------------------------------------------

def test_tfidf_unrelated_query_returns_empty(real_index, real_chunks):
    """Completely off-topic query returns an empty list from both TF-IDF and keyword fallback."""
    results = retrieve_tfidf("ancient Roman history", real_index, top_k=3)
    assert results == [], (
        f"Expected empty list for off-topic query, got {len(results)} result(s)"
    )


# ---------------------------------------------------------------------------
# Result schema tests
# ---------------------------------------------------------------------------

def test_tfidf_result_has_required_metadata(real_index):
    """Each result dict contains all required metadata fields."""
    results = retrieve_tfidf("What are the password requirements?", real_index, top_k=3)
    assert len(results) >= 1
    for result in results:
        assert "source" in result, "Missing 'source' field"
        assert "chunk_id" in result, "Missing 'chunk_id' field"
        assert "text" in result, "Missing 'text' field"
        assert "score" in result, "Missing 'score' field"
        assert "retrieval_method" in result, "Missing 'retrieval_method' field"


def test_tfidf_retrieval_method_is_tfidf(real_index):
    """All TF-IDF results carry retrieval_method = 'tfidf'."""
    results = retrieve_tfidf("What are the password requirements?", real_index, top_k=3)
    assert len(results) >= 1
    for result in results:
        assert result["retrieval_method"] == "tfidf", (
            f"Expected retrieval_method='tfidf', got '{result['retrieval_method']}'"
        )


def test_tfidf_score_is_between_zero_and_one(real_index):
    """All TF-IDF cosine similarity scores are valid floats in [0.0, 1.0]."""
    results = retrieve_tfidf("What are the password requirements?", real_index, top_k=3)
    assert len(results) >= 1
    for result in results:
        score = result["score"]
        assert isinstance(score, float), f"Score should be a float, got {type(score)}"
        assert 0.0 <= score <= 1.0, f"Score {score} is outside the [0.0, 1.0] range"


def test_tfidf_top_k_respected(real_index):
    """retrieve_tfidf() returns at most top_k results."""
    results = retrieve_tfidf("What are the password requirements?", real_index, top_k=2)
    assert len(results) <= 2, f"Expected at most 2 results, got {len(results)}"


# ---------------------------------------------------------------------------
# Keyword fallback test (controlled)
# ---------------------------------------------------------------------------

def test_tfidf_keyword_fallback_returns_keyword_method(real_chunks):
    """When min_score=0.99 forces TF-IDF to find nothing, the keyword fallback fires.

    This test exercises the fallback path in a controlled way:
      - min_score=0.99 is unreachably high for any real TF-IDF cosine similarity score.
      - The query "password security" has meaningful keyword overlap with security_guidelines.txt.
      - So keyword retrieval should find it and return retrieval_method="keyword".
    """
    # Build a fresh index (not the module-level one, to use a custom min_score)
    index = prepare_tfidf_index(real_chunks)
    results = retrieve_tfidf("password security", index, top_k=3, min_score=0.99)

    assert len(results) >= 1, (
        "Expected keyword fallback to return results when TF-IDF threshold is 0.99"
    )
    for result in results:
        assert result["retrieval_method"] == "keyword", (
            f"Expected retrieval_method='keyword' for fallback results, "
            f"got '{result['retrieval_method']}'"
        )
        assert result["score"] == 0, (
            f"Expected score=0 for keyword fallback results, got {result['score']}"
        )
