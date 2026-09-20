"""
Enterprise RAG Assistant - Main Streamlit Entry Point

Run locally with:
    streamlit run app/main.py
"""

import sys
from pathlib import Path

# Add project root directory to sys.path to ensure `app` package imports work cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from app import config
from app.rag import ingest_documents, chunk_documents, prepare_tfidf_index, retrieve_tfidf, generate_answer


def setup_page_configuration() -> None:
    """Configures page title, icon, and layout."""
    st.set_page_config(
        page_title=config.APP_TITLE,
        page_icon="🤖",
        layout="wide"
    )


@st.cache_data
def load_and_process_documents() -> tuple[list[dict], list[dict]]:
    """Loads sample documents from disk and splits them into text chunks.

    Returns:
        tuple[list[dict], list[dict]]: (raw_documents, chunks)
    """
    documents = ingest_documents("data/sample_docs")
    chunks = chunk_documents(documents, chunk_size=config.DEFAULT_CHUNK_SIZE, overlap=config.DEFAULT_CHUNK_OVERLAP)
    return documents, chunks


@st.cache_resource
def build_tfidf_index() -> dict:
    """Build and cache the TF-IDF index from document chunks.

    INDEXING PHASE -- called once per session, not on every query.

    Uses @st.cache_resource because the index contains stateful objects:
    a fitted TfidfVectorizer and a sparse corpus matrix.  These are
    model/index resources, not plain serialisable data.

    Internally calls load_and_process_documents() (itself cached via
    @st.cache_data) so no unhashable arguments need to be passed in.

    Returns:
        dict: Prepared TF-IDF index with vectorizer, corpus_matrix, and chunks.
    """
    _, chunks = load_and_process_documents()
    return prepare_tfidf_index(chunks)


def render_sidebar(doc_count: int, chunk_count: int) -> None:
    """Renders project status, document metrics, and pipeline stage in sidebar."""
    st.sidebar.title("📊 Pipeline Status")

    st.sidebar.metric(label="📄 Local Documents Loaded", value=doc_count)
    st.sidebar.metric(label="🧩 Chunks Generated", value=chunk_count)

    st.sidebar.divider()

    st.sidebar.info(
        "**Current Pipeline Stage:**\n\n"
        "⚡ **Local RAG Prototype**\n\n"
        "• Local text ingestion & window chunking\n"
        "• 🔍 **Retrieval Mode: TF-IDF Vector Similarity**\n"
        "• Keyword retrieval fallback\n"
        "• Grounded local answer synthesis"
    )

    st.sidebar.subheader("Planned Learning Stack")
    st.sidebar.markdown(
        """
        - **Language & UI:** Python, Streamlit
        - **RAG Core:** Embeddings, LLM, Retrieval Pipeline
        - **Containers & Cloud:** Docker, Google Cloud Platform (GCP)
        - **Compute & Storage:** Cloud Run, Cloud Storage, BigQuery
        - **Registry & DevOps:** Artifact Registry, GitHub Actions, Terraform
        """
    )


def render_main_content(chunks: list[dict], tfidf_index: dict) -> None:
    """Renders main interface, chat interaction, and retrieved context expander."""
    st.title(config.APP_TITLE)

    st.markdown(
        """
        Welcome to the **Enterprise RAG Assistant**.

        This local prototype demonstrates document ingestion, chunking, TF-IDF vector
        retrieval, and grounded answer generation using sample enterprise documents
        (security guidelines, HR policies, and product FAQs).
        """
    )

    st.divider()

    st.subheader("💬 Ask a Question")
    st.caption(
        "Try asking: *'What are the password requirements?'*, "
        "*'How many days of annual leave do I get?'*, or "
        "*'What are the API rate limits?'*"
    )

    # Chat input field
    user_query = st.chat_input("Ask a question about local sample documents...")

    if user_query:
        # Display user query
        with st.chat_message("user"):
            st.write(user_query)

        # QUERY-TIME: transform only the query, compare against pre-built index
        retrieved_chunks = retrieve_tfidf(user_query, tfidf_index, top_k=3)
        answer = generate_answer(user_query, retrieved_chunks)

        # Display assistant answer
        with st.chat_message("assistant"):
            st.markdown(answer)

            # Display retrieved sources and chunks in expandable section
            if retrieved_chunks:
                with st.expander("📚 View Retrieved Sources & Context Chunks"):
                    for idx, chunk in enumerate(retrieved_chunks, start=1):
                        method = chunk.get("retrieval_method", "unknown")
                        score = chunk.get("score", 0)
                        score_display = f"{score:.4f}" if method == "tfidf" else "n/a (keyword)"
                        st.markdown(
                            f"**Chunk #{idx}** — "
                            f"**Source:** `{chunk.get('source')}` | "
                            f"**Chunk ID:** `{chunk.get('chunk_id')}` | "
                            f"**Method:** `{method}` | "
                            f"**Score:** `{score_display}`"
                        )
                        st.code(chunk.get("text", ""), language="text")


def main() -> None:
    """Main application execution pipeline."""
    setup_page_configuration()
    documents, chunks = load_and_process_documents()

    # INDEXING PHASE: build the TF-IDF index once and cache it for the session.
    # The index is built from the same cached chunks, so this is always consistent.
    tfidf_index = build_tfidf_index()

    render_sidebar(len(documents), len(chunks))
    render_main_content(chunks, tfidf_index)


if __name__ == "__main__":
    main()
