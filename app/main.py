"""
Enterprise RAG Assistant - Main Streamlit Entry Point

Run locally with:
    streamlit run app/main.py
"""

import logging
import sys
from pathlib import Path

# Add project root directory to sys.path to ensure `app` package imports work cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from app import config
from app.rag import (
    ingest_documents,
    chunk_documents,
    prepare_tfidf_index,
    retrieve_tfidf,
    prepare_vertex_index,
    retrieve_vertex,
    create_vertex_client,
    generate_answer,
)


def setup_page_configuration() -> None:
    """Configures page title, icon, and layout."""
    st.set_page_config(
        page_title=config.APP_TITLE,
        page_icon="🤖",
        layout="wide",
    )


@st.cache_data
def load_and_process_documents() -> tuple[list[dict], list[dict]]:
    """Loads sample documents from disk and splits them into text chunks.

    Returns:
        tuple[list[dict], list[dict]]: (raw_documents, chunks)
    """
    documents = ingest_documents("data/sample_docs")
    chunks = chunk_documents(
        documents,
        chunk_size=config.DEFAULT_CHUNK_SIZE,
        overlap=config.DEFAULT_CHUNK_OVERLAP,
    )
    return documents, chunks


@st.cache_resource
def get_vertex_client():
    """Create and cache the Vertex AI Gen AI client.

    Uses @st.cache_resource so the client object is reused across all
    queries and Streamlit reruns without reconnecting on every interaction.

    Returns:
        A configured Vertex AI client, or None if Vertex AI is not configured
        or the client cannot be initialised.
    """
    if not config.VERTEX_ENABLED:
        return None
    try:
        return create_vertex_client()
    except Exception as exc:
        logging.warning("[Vertex] Failed to create client: %s", exc)
        return None


@st.cache_resource
def build_vertex_index(chunks: list[dict]) -> dict | None:
    """Build and cache the Vertex AI embedding index from document chunks.

    INDEXING PHASE -- embeds document chunks once using Vertex AI and caches
    the result as a reusable Streamlit resource.  The cache is keyed on the
    chunks argument, so the index is rebuilt automatically when documents
    change, but reused across all queries while documents remain the same.

    Args:
        chunks: The current list of document chunk dicts.

    Returns:
        dict: Prepared Vertex index with embeddings and chunk metadata, or None
            if Vertex AI is not configured or the indexing call fails.
    """
    client = get_vertex_client()
    if client is None:
        return None
    try:
        return prepare_vertex_index(chunks, client)
    except Exception as exc:
        logging.warning(
            "[Vertex] Failed to build embedding index: %s. "
            "Falling back to TF-IDF retrieval.",
            exc,
        )
        return None


@st.cache_resource
def build_tfidf_index() -> dict:
    """Build and cache the TF-IDF index from document chunks.

    INDEXING PHASE -- called once per session, not on every query.

    Returns:
        dict: Prepared TF-IDF index with vectorizer, corpus_matrix, and chunks.
    """
    _, chunks = load_and_process_documents()
    return prepare_tfidf_index(chunks)


def _retrieve(
    query: str,
    vertex_index: dict | None,
    tfidf_index: dict,
    top_k: int = 3,
) -> list[dict]:
    """Three-tier retrieval with automatic fallback.

    Retrieval order:
        1. Vertex AI semantic retrieval (primary)
           -- If Vertex returns results above VERTEX_MIN_SCORE, return them.
           -- If Vertex returns [] (all chunks below threshold) OR raises an
              exception, fall through to Tier 2.  Rejected low-confidence
              chunks are never returned to the caller.
        2. TF-IDF vector retrieval (secondary)
           -- If TF-IDF finds results above its threshold, return them.
           -- If TF-IDF finds nothing, its internal keyword fallback runs.
        3. Keyword overlap retrieval (tertiary, runs inside retrieve_tfidf)
        4. [] -- if no retrieval method finds meaningful context.

    This means a query like "Who was Julius Caesar?" (top Vertex score 0.5092,
    below the 0.6 threshold) will:
        - be rejected by Vertex  -> []
        - fall through to TF-IDF -> [] (no lexical overlap)
        - keyword fallback runs  -> [] (no keyword match)
        - caller receives []     -> no-answer response shown, no context expander

    A synonym query like "How much vacation time do employees get?"
    (top Vertex score 0.7275, above 0.6) will:
        - be accepted by Vertex  -> hr_policy.txt chunk returned
        - TF-IDF is not invoked

    Args:
        query:        The user question or search string.
        vertex_index: The prepared Vertex index, or None when Vertex is disabled.
        tfidf_index:  The prepared TF-IDF index (always available).
        top_k:        Maximum number of results to return.

    Returns:
        list[dict]: Retrieved chunks with consistent result schema, or [] when
            no retrieval method finds a relevant match above its threshold.
    """
    # Tier 1: Vertex AI semantic retrieval
    if vertex_index is not None:
        client = get_vertex_client()
        if client is not None:
            try:
                vertex_results = retrieve_vertex(
                    query,
                    vertex_index,
                    client,
                    top_k=top_k,
                    min_score=config.VERTEX_MIN_SCORE,
                )
                if vertex_results:
                    # At least one chunk exceeded the confidence threshold.
                    return vertex_results
                # Vertex ran successfully but all chunks scored below the
                # threshold.  Fall through to TF-IDF -- do NOT return [] here,
                # because TF-IDF or keyword may still find relevant content.
                logging.debug(
                    "[Vertex] No chunk above threshold (%.2f) for query: %r. "
                    "Falling back to TF-IDF.",
                    config.VERTEX_MIN_SCORE,
                    query[:80],
                )
            except Exception as exc:
                logging.warning(
                    "[Vertex] Query-time retrieval failed: %s. "
                    "Falling back to TF-IDF.",
                    exc,
                )

    # Tier 2 + 3: TF-IDF with internal keyword fallback
    return retrieve_tfidf(query, tfidf_index, top_k=top_k)


def render_sidebar(doc_count: int, chunk_count: int, vertex_active: bool) -> None:
    """Renders pipeline status, document metrics, and technology stack in sidebar."""
    st.sidebar.title("Pipeline Status")

    st.sidebar.metric(label="Documents Indexed", value=doc_count)
    st.sidebar.metric(label="Chunks Indexed", value=chunk_count)

    st.sidebar.divider()

    # Retrieval architecture
    st.sidebar.subheader("Retrieval Architecture")
    vertex_badge = "🟢 Active" if vertex_active else "🔴 Disabled (no credentials)"
    st.sidebar.markdown(
        f"""
| Layer | Method | Status |
|---|---|---|
| Primary | Vertex AI Semantic | {vertex_badge} |
| Secondary | TF-IDF Vector | Always available |
| Fallback | Keyword Overlap | Always available |
"""
    )

    st.sidebar.divider()

    # Current stack
    st.sidebar.subheader("Current Capabilities")
    st.sidebar.markdown(
        """
- Python 3.13, Streamlit
- Local document ingestion & chunking
- Vertex AI semantic embeddings (`gemini-embedding-001`)
- TF-IDF vector retrieval + cosine similarity
- Keyword overlap retrieval fallback
- Grounded local answer extraction
- Automated test suite (pytest)
        """
    )

    st.sidebar.divider()

    # Planned next
    st.sidebar.subheader("Planned Next")
    st.sidebar.markdown(
        """
- LLM-grounded generation (Vertex AI / Gemini)
- Docker containerisation
- Cloud Run deployment
- Cloud Storage document store
- BigQuery analytics
- Artifact Registry & GitHub Actions CI/CD
- Terraform infrastructure-as-code
        """
    )


def render_main_content(
    chunks: list[dict],
    tfidf_index: dict,
    vertex_index: dict | None,
) -> None:
    """Renders main interface, chat interaction, and retrieved context expander."""
    st.title(config.APP_TITLE)

    if vertex_index is not None:
        retrieval_caption = (
            "Vertex AI semantic embeddings · TF-IDF vector fallback · "
            "Keyword fallback · Grounded local extraction"
        )
    else:
        retrieval_caption = (
            "TF-IDF vector retrieval · Keyword fallback · "
            "Grounded local extraction"
        )

    st.caption(retrieval_caption)

    st.markdown(
        """
        Ask a question and the assistant will retrieve the most relevant passages
        from the knowledge base and extract a grounded answer.

        The knowledge base contains fictional sample documents on HR policies,
        security guidelines, and product specifications — designed to demonstrate
        retrieval across diverse enterprise document types.
        """
    )

    st.divider()

    st.subheader("Ask a Question")
    st.caption(
        "Examples: *'What are the password requirements?'* · "
        "*'How many days of annual leave do I get?'* · "
        "*'What are the API rate limits?'*"
    )

    user_query = st.chat_input("Ask a question about the knowledge base...")

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        # QUERY-TIME: three-tier retrieval (Vertex -> TF-IDF -> keyword -> [])
        retrieved_chunks = _retrieve(user_query, vertex_index, tfidf_index, top_k=3)
        answer = generate_answer(user_query, retrieved_chunks)

        with st.chat_message("assistant"):
            st.markdown(answer)

            # Determine generation method label for the caption
            if retrieved_chunks:
                method = retrieved_chunks[0].get("retrieval_method", "unknown")
                method_label = {
                    "vertex": "Vertex AI Semantic",
                    "tfidf": "TF-IDF Lexical",
                    "keyword": "Keyword Overlap",
                }.get(method, method)
                st.caption(
                    f"Retrieval: {method_label} · Generation: Local grounded extraction"
                )

            if retrieved_chunks:
                with st.expander("View Retrieved Sources & Context"):
                    for idx, chunk in enumerate(retrieved_chunks, start=1):
                        method = chunk.get("retrieval_method", "unknown")
                        score = chunk.get("score", 0)

                        if method in ("vertex", "tfidf"):
                            score_display = f"{score:.4f}"
                        else:
                            score_display = "n/a (keyword match)"

                        method_label = {
                            "vertex": "Vertex AI Semantic",
                            "tfidf": "TF-IDF Lexical",
                            "keyword": "Keyword Overlap",
                        }.get(method, method)

                        st.markdown(
                            f"**#{idx}** &nbsp;·&nbsp; "
                            f"**Source:** `{chunk.get('source')}` &nbsp;·&nbsp; "
                            f"**Chunk:** `{chunk.get('chunk_id')}` &nbsp;·&nbsp; "
                            f"**Method:** {method_label} &nbsp;·&nbsp; "
                            f"**Score:** `{score_display}`"
                        )
                        st.code(chunk.get("text", ""), language="text")


def main() -> None:
    """Main application execution pipeline."""
    setup_page_configuration()
    documents, chunks = load_and_process_documents()

    # INDEXING PHASE: build indexes once and cache them as reusable resources.
    vertex_index = build_vertex_index(chunks)
    tfidf_index = build_tfidf_index()

    render_sidebar(len(documents), len(chunks), vertex_active=vertex_index is not None)
    render_main_content(chunks, tfidf_index, vertex_index)


if __name__ == "__main__":
    main()
