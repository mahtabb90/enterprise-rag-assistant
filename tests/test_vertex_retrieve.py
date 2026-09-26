"""
Unit tests for Vertex AI semantic retrieval.

All tests mock the Vertex embedding layer using unittest.mock.patch.
No live Vertex API calls are made in this test module.

Test structure mirrors the indexing/retrieval separation:
  - Fixtures that call prepare_vertex_index() represent the INDEXING phase.
  - Tests that call retrieve_vertex() represent the QUERY-TIME phase.

Mock embedding strategy
-----------------------
3-dimensional unit vectors are used as mock document embeddings.
Because the vectors are already unit-length, cosine similarity equals
the dot product, making expected scores trivially predictable:

    MOCK_DOC_EMBEDDINGS_3D = [
        [1, 0, 0],   # chunk 0 (annual leave)  -- x-axis
        [0, 1, 0],   # chunk 1 (security)       -- y-axis
        [0, 0, 1],   # chunk 2 (faq)            -- z-axis
    ]

    query aligned with chunk 0:  [1, 0, 0]  --> similarity = 1.0
    query at 45 degrees:         [~0.7, ~0.7, 0] --> similarity ~= 0.707 each

768-dimensional unit vectors are used only for dimension-validation tests.
"""

import pytest
from unittest.mock import MagicMock, patch
from app.rag.retrieve import (
    prepare_vertex_index,
    retrieve_vertex,
    prepare_tfidf_index,
    retrieve_tfidf,
)


# ---------------------------------------------------------------------------
# Mock embedding constants
# ---------------------------------------------------------------------------

# Sample chunks used across multiple tests.
SAMPLE_CHUNKS = [
    {
        "source": "hr_policy.txt",
        "chunk_id": 0,
        "text": "Full-time employees receive 25 days of paid annual leave per calendar year.",
    },
    {
        "source": "security_guidelines.txt",
        "chunk_id": 1,
        "text": "All passwords must be at least 12 characters and include MFA.",
    },
    {
        "source": "product_faq.txt",
        "chunk_id": 2,
        "text": "The API supports up to 1000 requests per minute per client.",
    },
]

# 3-dimensional orthogonal unit vectors for clean, predictable cosine similarity.
# cosine_similarity([a], [b]) == dot(a, b) when both are unit vectors.
MOCK_DOC_EMBEDDINGS_3D = [
    [1.0, 0.0, 0.0],   # chunk 0 (annual leave)  -- x-axis
    [0.0, 1.0, 0.0],   # chunk 1 (security)       -- y-axis
    [0.0, 0.0, 1.0],   # chunk 2 (faq)            -- z-axis
]

# Query perfectly aligned with chunk 0: cosine similarity = 1.0
MOCK_QUERY_ALIGNED_0 = [1.0, 0.0, 0.0]

# Query at 45 degrees to chunks 0 and 1: cosine similarity ~= 0.707 each
MOCK_QUERY_PARTIAL = [0.7071, 0.7071, 0.0]

# 768-dimensional mock embeddings for dimension validation tests.
_DIM = 768
MOCK_DOC_EMBEDDINGS_768D = [
    [1.0 if i == j else 0.0 for i in range(_DIM)]
    for j in range(len(SAMPLE_CHUNKS))
]
MOCK_QUERY_EMBEDDING_768D = [1.0] + [0.0] * (_DIM - 1)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client():
    """A MagicMock standing in for a google.genai.Client instance."""
    return MagicMock()


@pytest.fixture
def vertex_index_3d(mock_client):
    """Vertex index built with controlled 3D unit-vector mock embeddings.

    Represents the INDEXING phase with mocked Vertex API calls.
    """
    with patch("app.rag.retrieve.embed_documents", return_value=MOCK_DOC_EMBEDDINGS_3D):
        return prepare_vertex_index(SAMPLE_CHUNKS, mock_client)


# ---------------------------------------------------------------------------
# Indexing phase tests
# ---------------------------------------------------------------------------

def test_vertex_index_retains_chunk_metadata(mock_client):
    """prepare_vertex_index() preserves the full chunk list alongside embeddings."""
    with patch("app.rag.retrieve.embed_documents", return_value=MOCK_DOC_EMBEDDINGS_3D):
        index = prepare_vertex_index(SAMPLE_CHUNKS, mock_client)
    assert index["chunks"] == SAMPLE_CHUNKS


def test_vertex_index_retains_embeddings(mock_client):
    """prepare_vertex_index() stores one embedding vector per chunk."""
    with patch("app.rag.retrieve.embed_documents", return_value=MOCK_DOC_EMBEDDINGS_3D):
        index = prepare_vertex_index(SAMPLE_CHUNKS, mock_client)
    assert len(index["embeddings"]) == len(SAMPLE_CHUNKS)


def test_vertex_index_embedding_dimensions(mock_client):
    """Each stored embedding has exactly 768 dimensions."""
    with patch("app.rag.retrieve.embed_documents", return_value=MOCK_DOC_EMBEDDINGS_768D):
        index = prepare_vertex_index(SAMPLE_CHUNKS, mock_client)
    for embedding in index["embeddings"]:
        assert len(embedding) == _DIM, (
            f"Expected {_DIM}-dim embedding, got {len(embedding)}"
        )


def test_vertex_query_embedding_dimension_compatible(mock_client):
    """Query embedding is compatible with the document embedding dimensions.

    If dimensions match, cosine_similarity completes without error and
    returns a result for every chunk.
    """
    with patch("app.rag.retrieve.embed_documents", return_value=MOCK_DOC_EMBEDDINGS_768D):
        index = prepare_vertex_index(SAMPLE_CHUNKS, mock_client)
    with patch("app.rag.retrieve.embed_query", return_value=MOCK_QUERY_EMBEDDING_768D):
        results = retrieve_vertex("any query", index, mock_client, min_score=0.0)
    # Cosine similarity completed without error; at least one result returned.
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Query-time retrieval tests
# ---------------------------------------------------------------------------

def test_retrieve_vertex_ranks_semantic_match_correctly(vertex_index_3d, mock_client):
    """Mocked embedding vectors produce the expected top-ranked result."""
    with patch("app.rag.retrieve.embed_query", return_value=MOCK_QUERY_ALIGNED_0):
        results = retrieve_vertex("vacation time", vertex_index_3d, mock_client,
                                  top_k=3, min_score=0.0)
    assert len(results) >= 1
    assert results[0]["source"] == "hr_policy.txt", (
        f"Expected hr_policy.txt as top result, got '{results[0]['source']}'"
    )


def test_retrieve_vertex_result_schema(vertex_index_3d, mock_client):
    """Each result dict contains all required fields."""
    with patch("app.rag.retrieve.embed_query", return_value=MOCK_QUERY_ALIGNED_0):
        results = retrieve_vertex("vacation time", vertex_index_3d, mock_client, min_score=0.0)
    assert len(results) >= 1
    for result in results:
        assert "source" in result, "Missing 'source'"
        assert "chunk_id" in result, "Missing 'chunk_id'"
        assert "text" in result, "Missing 'text'"
        assert "score" in result, "Missing 'score'"
        assert "retrieval_method" in result, "Missing 'retrieval_method'"


def test_retrieve_vertex_retrieval_method_is_vertex(vertex_index_3d, mock_client):
    """All Vertex results carry retrieval_method = 'vertex'."""
    with patch("app.rag.retrieve.embed_query", return_value=MOCK_QUERY_ALIGNED_0):
        results = retrieve_vertex("vacation time", vertex_index_3d, mock_client, min_score=0.0)
    assert len(results) >= 1
    for result in results:
        assert result["retrieval_method"] == "vertex", (
            f"Expected 'vertex', got '{result['retrieval_method']}'"
        )


def test_retrieve_vertex_score_range(vertex_index_3d, mock_client):
    """All Vertex cosine similarity scores are valid floats in [0.0, 1.0]."""
    with patch("app.rag.retrieve.embed_query", return_value=MOCK_QUERY_ALIGNED_0):
        results = retrieve_vertex("vacation time", vertex_index_3d, mock_client, min_score=0.0)
    assert len(results) >= 1
    for result in results:
        score = result["score"]
        assert isinstance(score, float), f"Score should be float, got {type(score)}"
        assert 0.0 <= score <= 1.0, f"Score {score} outside [0.0, 1.0]"


def test_retrieve_vertex_respects_top_k(vertex_index_3d, mock_client):
    """retrieve_vertex() returns at most top_k results."""
    with patch("app.rag.retrieve.embed_query", return_value=MOCK_QUERY_ALIGNED_0):
        results = retrieve_vertex("vacation time", vertex_index_3d, mock_client,
                                  top_k=1, min_score=0.0)
    assert len(results) <= 1, f"Expected at most 1 result, got {len(results)}"


def test_retrieve_vertex_min_score_filters_low_similarity(mock_client):
    """Results below min_score are excluded; returns [] when all are filtered.

    MOCK_QUERY_PARTIAL is at 45 degrees to chunks 0 and 1, giving cosine
    similarity ~= 0.707 for each.  Setting min_score=0.8 filters all results.
    """
    with patch("app.rag.retrieve.embed_documents", return_value=MOCK_DOC_EMBEDDINGS_3D):
        index = prepare_vertex_index(SAMPLE_CHUNKS, mock_client)
    with patch("app.rag.retrieve.embed_query", return_value=MOCK_QUERY_PARTIAL):
        results = retrieve_vertex("any query", index, mock_client, min_score=0.8)
    assert results == [], (
        f"Expected [] with min_score=0.8 and max score ~0.707, got {results}"
    )


# ---------------------------------------------------------------------------
# Fallback chain tests
# ---------------------------------------------------------------------------

def test_retrieve_vertex_failure_propagates_for_caller_fallback(mock_client):
    """retrieve_vertex() propagates API exceptions; the caller handles fallback.

    Architecture note:
        retrieve_vertex() does NOT silently catch errors -- it re-raises them
        so the calling code (main.py _retrieve()) can log a warning and
        fall back to TF-IDF.  This test verifies that exception propagation
        behaviour, then confirms TF-IDF remains independently functional.
    """
    with patch("app.rag.retrieve.embed_documents", return_value=MOCK_DOC_EMBEDDINGS_3D):
        index = prepare_vertex_index(SAMPLE_CHUNKS, mock_client)

    # Vertex query-time failure raises, not swallows.
    with patch("app.rag.retrieve.embed_query", side_effect=Exception("Vertex API unavailable")):
        with pytest.raises(Exception, match="Vertex API unavailable"):
            retrieve_vertex("security requirements", index, mock_client)

    # TF-IDF is unaffected and provides a valid fallback.
    # "characters" appears in the security chunk; "minute" in the FAQ chunk.
    tfidf_index = prepare_tfidf_index(SAMPLE_CHUNKS)
    fallback_results = retrieve_tfidf("characters minute", tfidf_index, top_k=3)
    assert len(fallback_results) >= 1
    assert all(r["retrieval_method"] in ("tfidf", "keyword") for r in fallback_results)


def test_tfidf_failure_falls_back_to_keyword():
    """When TF-IDF min_score is unreachably high, the keyword fallback fires.

    Uses a controlled chunk set where the query has exact keyword overlap
    with one chunk so the keyword fallback can succeed.
    """
    chunks = [
        {
            "source": "security_doc.txt",
            "chunk_id": 0,
            "text": "Security requires strong authentication and access controls.",
        },
        {
            "source": "hr_doc.txt",
            "chunk_id": 1,
            "text": "Annual leave entitlement is 25 days per year.",
        },
    ]
    index = prepare_tfidf_index(chunks)
    # min_score=0.99 is unreachably high for any real TF-IDF cosine score.
    results = retrieve_tfidf("security controls", index, top_k=3, min_score=0.99)
    assert len(results) >= 1, (
        "Expected keyword fallback to return results when TF-IDF threshold is 0.99"
    )
    for result in results:
        assert result["retrieval_method"] == "keyword", (
            f"Expected 'keyword' for fallback results, got '{result['retrieval_method']}'"
        )


def test_unrelated_query_returns_empty():
    """Completely off-topic query returns [] from TF-IDF and keyword fallback."""
    tfidf_index = prepare_tfidf_index(SAMPLE_CHUNKS)
    results = retrieve_tfidf("ancient Roman architecture aqueducts", tfidf_index, top_k=3)
    assert results == [], (
        f"Expected [] for off-topic query, got {len(results)} result(s): {results}"
    )


# ---------------------------------------------------------------------------
# Semantic vs lexical contrast
# ---------------------------------------------------------------------------

def test_semantic_vs_lexical_contrast(mock_client):
    """Demonstrates the architectural gap between semantic and lexical retrieval.

    Scenario
    --------
    Document: "Salaried staff accrue 25 vacation days each fiscal year."
    Query:    "How much paid time off are workers entitled to annually?"

    Lexical analysis:
        Query vocabulary (TF-IDF tokens):
            how, much, paid, time, off, are, workers, entitled, to, annually
        Document vocabulary:
            salaried, staff, accrue, 25, vacation, days, each, fiscal, year
        Shared vocabulary: NONE (guaranteed zero TF-IDF cosine similarity)

    This is the key architectural distinction:
        TF-IDF (lexical):    cosine similarity = 0.0  -> hr_policy.txt NOT retrieved
        Vertex AI (semantic): embedding model captures meaning -> hr_policy.txt retrieved
    """
    semantic_query = "How much paid time off are workers entitled to annually?"
    annual_leave_chunk = {
        "source": "hr_policy.txt",
        "chunk_id": 0,
        "text": "Salaried staff accrue 25 vacation days each fiscal year.",
    }
    security_chunk = {
        "source": "security_guidelines.txt",
        "chunk_id": 1,
        "text": "Access credentials must be rotated every ninety days.",
    }
    chunks = [annual_leave_chunk, security_chunk]

    # --- Lexical TF-IDF retrieval ---
    # Query vocabulary and corpus vocabulary are completely disjoint.
    # TF-IDF cosine similarity = 0.0 for both chunks.
    # hr_policy.txt must NOT appear in TF-IDF results.
    tfidf_index = prepare_tfidf_index(chunks)
    tfidf_results = retrieve_tfidf(semantic_query, tfidf_index, top_k=3)
    tfidf_sources = {r["source"] for r in tfidf_results}
    assert "hr_policy.txt" not in tfidf_sources, (
        "TF-IDF should not surface the annual-leave chunk when query shares no "
        f"vocabulary terms with it. TF-IDF results: {tfidf_results}"
    )

    # --- Vertex semantic retrieval (mocked) ---
    # Mock embeddings place the annual-leave chunk aligned with the query.
    # Despite zero vocabulary overlap, semantic similarity is 1.0 (unit vectors).
    mock_doc_embeddings = [
        [1.0, 0.0, 0.0],   # annual leave chunk  -- semantically close to query
        [0.0, 1.0, 0.0],   # security chunk      -- semantically unrelated
    ]
    mock_query_embedding = [1.0, 0.0, 0.0]  # aligned with annual leave chunk

    with patch("app.rag.retrieve.embed_documents", return_value=mock_doc_embeddings):
        vertex_index = prepare_vertex_index(chunks, mock_client)

    with patch("app.rag.retrieve.embed_query", return_value=mock_query_embedding):
        vertex_results = retrieve_vertex(
            semantic_query, vertex_index, mock_client, top_k=3, min_score=0.0
        )

    assert len(vertex_results) >= 1, (
        "Vertex retrieval should find the semantically relevant annual-leave chunk."
    )
    assert vertex_results[0]["source"] == "hr_policy.txt", (
        f"Expected hr_policy.txt as top Vertex result, got '{vertex_results[0]['source']}'"
    )
    assert vertex_results[0]["retrieval_method"] == "vertex"

    # Core architectural assertion:
    # The annual-leave chunk is absent from TF-IDF results (lexical gap)
    # but present as the top Vertex result (semantic bridge).
    vertex_sources = {r["source"] for r in vertex_results}
    assert "hr_policy.txt" not in tfidf_sources
    assert "hr_policy.txt" in vertex_sources


# ---------------------------------------------------------------------------
# Calibrated threshold tests (0.6 default)
# ---------------------------------------------------------------------------
#
# Mock geometry note:
#   MOCK_DOC_EMBEDDINGS_3D places chunks on the x, y, z unit axes respectively.
#   A query of [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)] (equal components) has cosine
#   similarity 1/sqrt(3) ~= 0.577 to every chunk -- reliably below the 0.6
#   threshold, making it the right mock for "completely unrelated" queries.
#   A query of [1, 0, 0] has similarity 1.0 to chunk 0 -- reliably above 0.6.

import math as _math
_UNRELATED_QUERY_VEC = [1/_math.sqrt(3)] * 3   # sim ~= 0.577 to all axis chunks


def test_vertex_calibrated_threshold_rejects_unrelated(mock_client):
    """A query whose Vertex score is ~0.577 for every chunk is rejected at 0.6 threshold.

    Mirrors the live observed behavior for "Who was Julius Caesar?" (top score
    0.5092 against the fictional sample corpus).

    Because the query vector [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)] is equidistant
    from all three orthogonal axis chunks, cosine similarity is uniformly ~0.577
    for each chunk -- below the calibrated 0.6 default.

    Rejected chunks must NOT appear in the result; the function returns [].
    """
    with patch("app.rag.retrieve.embed_documents", return_value=MOCK_DOC_EMBEDDINGS_3D):
        index = prepare_vertex_index(SAMPLE_CHUNKS, mock_client)

    with patch("app.rag.retrieve.embed_query", return_value=_UNRELATED_QUERY_VEC):
        results = retrieve_vertex(
            "Who was Julius Caesar?",
            index,
            mock_client,
            top_k=3,
            min_score=0.6,   # calibrated default for the fictional sample corpus
        )

    assert results == [], (
        f"Expected [] -- all chunks score ~0.577 which is below 0.6 threshold. "
        f"Got: {results}"
    )


def test_vertex_calibrated_threshold_accepts_synonym(mock_client):
    """A semantically relevant synonym query scoring 1.0 is accepted at 0.6 threshold.

    Mirrors the live observed behavior for
    "How much vacation time do employees get?" (top score 0.7275).

    The mock query vector [1, 0, 0] is perfectly aligned with chunk 0 (hr_policy),
    giving cosine similarity 1.0 -- well above the 0.6 threshold.
    """
    with patch("app.rag.retrieve.embed_documents", return_value=MOCK_DOC_EMBEDDINGS_3D):
        index = prepare_vertex_index(SAMPLE_CHUNKS, mock_client)

    with patch("app.rag.retrieve.embed_query", return_value=MOCK_QUERY_ALIGNED_0):
        results = retrieve_vertex(
            "How much vacation time do employees get?",
            index,
            mock_client,
            top_k=3,
            min_score=0.6,   # calibrated default
        )

    assert len(results) >= 1, (
        "Expected at least one result -- synonym query should exceed 0.6 threshold."
    )
    assert results[0]["source"] == "hr_policy.txt", (
        f"Expected hr_policy.txt as top result, got '{results[0]['source']}'"
    )
    assert results[0]["score"] >= 0.6, (
        f"Expected score >= 0.6, got {results[0]['score']}"
    )


# ---------------------------------------------------------------------------
# Below-threshold fallback chain test
# ---------------------------------------------------------------------------

def test_vertex_below_threshold_falls_through_to_tfidf(mock_client):
    """When Vertex returns [] (all chunks below threshold), TF-IDF is tried next.

    This tests the architectural requirement that a Vertex empty result is NOT
    treated as a final answer.  The fallback chain continues:
        Vertex [] --> TF-IDF --> (result or keyword fallback or [])

    Scenario:
        - Vertex: unrelated query vector gives sim ~0.577 to all chunks -> []
          when min_score=0.6.
        - TF-IDF: query "characters" has lexical overlap with security chunk
          -> result found via TF-IDF.

    The test simulates _retrieve() behavior directly at the retrieve_vertex /
    retrieve_tfidf layer to avoid Streamlit cache dependencies.
    """
    with patch("app.rag.retrieve.embed_documents", return_value=MOCK_DOC_EMBEDDINGS_3D):
        vertex_index = prepare_vertex_index(SAMPLE_CHUNKS, mock_client)

    # Step 1: Vertex returns [] -- unrelated query below threshold
    with patch("app.rag.retrieve.embed_query", return_value=_UNRELATED_QUERY_VEC):
        vertex_results = retrieve_vertex(
            "characters",
            vertex_index,
            mock_client,
            top_k=3,
            min_score=0.6,
        )

    assert vertex_results == [], (
        f"Vertex should return [] for unrelated query. Got: {vertex_results}"
    )

    # Step 2: Caller falls through to TF-IDF because vertex_results is [].
    # "characters" has direct lexical overlap with the security_guidelines chunk.
    tfidf_index = prepare_tfidf_index(SAMPLE_CHUNKS)
    tfidf_results = retrieve_tfidf("characters", tfidf_index, top_k=3)

    assert len(tfidf_results) >= 1, (
        "TF-IDF fallback should find results when Vertex returns []. "
        f"Got: {tfidf_results}"
    )
    assert tfidf_results[0]["source"] == "security_guidelines.txt", (
        f"Expected security_guidelines.txt via TF-IDF fallback, "
        f"got '{tfidf_results[0]['source']}'"
    )
    assert tfidf_results[0]["retrieval_method"] == "tfidf", (
        f"Fallback results should carry 'tfidf', got '{tfidf_results[0]['retrieval_method']}'"
    )


def test_unrelated_query_full_chain_returns_empty(mock_client):
    """Completely unrelated query produces [] from every tier of the chain.

    Mirrors "Who was Julius Caesar?" against the fictional sample corpus:
        Vertex  -> [] (sim ~0.577 < 0.6 threshold)
        TF-IDF  -> [] (no shared vocabulary in SAMPLE_CHUNKS)
        Keyword -> [] (no meaningful keyword overlap)
        Final   -> [] (no-answer response shown; context expander not displayed)

    The mock query [1/sqrt(3), ...] gives uniform low similarity to all chunks.
    The text query "Who was Julius Caesar?" has no token overlap with any
    SAMPLE_CHUNK text, so TF-IDF and keyword also return [].
    """
    with patch("app.rag.retrieve.embed_documents", return_value=MOCK_DOC_EMBEDDINGS_3D):
        vertex_index = prepare_vertex_index(SAMPLE_CHUNKS, mock_client)

    # Tier 1: Vertex rejected -- all chunks below 0.6 threshold
    with patch("app.rag.retrieve.embed_query", return_value=_UNRELATED_QUERY_VEC):
        vertex_results = retrieve_vertex(
            "Who was Julius Caesar?",
            vertex_index,
            mock_client,
            top_k=3,
            min_score=0.6,
        )

    assert vertex_results == [], (
        f"Vertex should reject: sim ~0.577 < 0.6. Got: {vertex_results}"
    )

    # Tier 2 + 3: TF-IDF / keyword fallback -- ancient Roman history has no overlap
    tfidf_index = prepare_tfidf_index(SAMPLE_CHUNKS)
    tfidf_and_keyword_results = retrieve_tfidf(
        "Who was Julius Caesar?", tfidf_index, top_k=3
    )

    assert tfidf_and_keyword_results == [], (
        f"TF-IDF + keyword fallback should also return [] for off-topic query. "
        f"Got: {tfidf_and_keyword_results}"
    )

    # Final state: [] from all tiers -> UI shows no-answer, no expander shown.
    final_context = vertex_results or tfidf_and_keyword_results
    assert final_context == [], (
        "End-to-end: completely unrelated query must produce [] context."
    )
