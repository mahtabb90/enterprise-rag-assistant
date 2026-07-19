"""
Unit tests for the local answer generation module.
"""

from app.rag.ingest import ingest_documents
from app.rag.chunker import chunk_documents
from app.rag.retrieve import retrieve_context
from app.rag.generate import generate_answer, _extract_sentences, _score_sentence


def test_generate_answer_no_context():
    """Tests that a query returning no context returns the helpful fallback message."""
    query = "What is quantum physics?"
    answer = generate_answer(query, context=[])

    assert "I could not find an answer to your question" in answer
    assert "password requirements" in answer
    assert "annual leave policies" in answer
    assert "API rate limits" in answer


def test_generate_answer_company_revenue_fallback():
    """Tests that company revenue query returns fallback message as no answer is found."""
    docs = ingest_documents("data/sample_docs")
    chunks = chunk_documents(docs)

    query = "What is the company revenue?"
    retrieved = retrieve_context(query, chunks)
    answer = generate_answer(query, retrieved)

    assert "I could not find an answer to your question" in answer


def test_generate_answer_annual_leave_query():
    """Tests that annual leave query includes 25 days and hr_policy.txt in source."""
    docs = ingest_documents("data/sample_docs")
    chunks = chunk_documents(docs)

    query = "How many days of annual leave do I get?"
    retrieved = retrieve_context(query, chunks)
    answer = generate_answer(query, retrieved)

    assert "25 days" in answer
    assert "hr_policy.txt" in answer
    assert "remote work" not in answer.lower()
    assert "Answer (Local Rule-Based Prototype)" in answer
    assert "Source:" in answer


def test_generate_answer_password_query():
    """Tests that password requirements query includes password content and security_guidelines.txt."""
    docs = ingest_documents("data/sample_docs")
    chunks = chunk_documents(docs)

    query = "What are the password requirements?"
    retrieved = retrieve_context(query, chunks)
    answer = generate_answer(query, retrieved)

    assert "password" in answer.lower()
    assert "12 characters" in answer
    assert "security_guidelines.txt" in answer
    assert "Source:" in answer


def test_generate_answer_api_rate_limits_query():
    """Tests that API rate limit query includes 1,000 requests per minute and product_faq.txt."""
    docs = ingest_documents("data/sample_docs")
    chunks = chunk_documents(docs)

    query = "What are the API rate limits?"
    retrieved = retrieve_context(query, chunks)
    answer = generate_answer(query, retrieved)

    assert "1,000 requests per minute" in answer
    assert "product_faq.txt" in answer
    assert "Source:" in answer


def test_extract_sentences_filters_headers():
    """Tests that header titles are filtered out and clean sentences are extracted."""
    text = (
        "3. Annual Leave and Paid Time Off (PTO)\n"
        "Full-time employees receive 25 days of paid annual leave per calendar year. "
        "Vacation requests should be submitted through the HR portal."
    )
    sentences = _extract_sentences(text)
    assert len(sentences) == 2
    assert "3. Annual Leave" not in sentences[0]
    assert sentences[0] == "Full-time employees receive 25 days of paid annual leave per calendar year."


def test_score_sentence_penalizes_remote_work_for_annual_leave():
    """Tests sentence scoring prioritizes annual leave sentence over remote work sentence."""
    query = "How many days of annual leave do I get?"
    remote_work_sent = "allowing up to two days per week of remote work"
    annual_leave_sent = "Full-time employees receive 25 days of paid annual leave per calendar year."

    remote_score = _score_sentence(remote_work_sent, query, rank=0)
    leave_score = _score_sentence(annual_leave_sent, query, rank=0)

    assert leave_score > remote_score

