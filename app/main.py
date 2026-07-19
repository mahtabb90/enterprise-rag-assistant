"""
Enterprise RAG Assistant - Main Streamlit Entry Point

Run locally with:
    streamlit run app/main.py
"""

import streamlit as st
import config
from app.rag import ingest_documents, chunk_documents, retrieve_context, generate_answer


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


def render_sidebar(doc_count: int, chunk_count: int) -> None:
    """Renders project status, document metrics, and pipeline stage in sidebar."""
    st.sidebar.title("📊 Pipeline Status")
    
    st.sidebar.metric(label="📄 Local Documents Loaded", value=doc_count)
    st.sidebar.metric(label="🧩 Chunks Generated", value=chunk_count)
    
    st.sidebar.divider()
    
    st.sidebar.info(
        "**Current Pipeline Stage:**\n\n"
        "⚡ **Local RAG Prototype (Day 2)**\n\n"
        "• Local text ingestion & window chunking\n"
        "• Normalized keyword similarity search\n"
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


def render_main_content(chunks: list[dict]) -> None:
    """Renders main interface, chat interaction, and retrieved context expander."""
    st.title(config.APP_TITLE)
    
    st.markdown(
        """
        Welcome to the **Enterprise RAG Assistant**.
        
        This local prototype demonstrates document ingestion, chunking, keyword retrieval, 
        and grounded answer generation using sample enterprise documents (security guidelines, 
        HR policies, and product FAQs).
        """
    )
    
    st.divider()
    
    st.subheader("💬 Ask a Question")
    st.caption("Try asking: *'What are the password requirements?'*, *'How many days of annual leave do I get?'*, or *'What are the API rate limits?'*")
    
    # Chat input field
    user_query = st.chat_input("Ask a question about local sample documents...")
    
    if user_query:
        # Display user query
        with st.chat_message("user"):
            st.write(user_query)
            
        # Retrieve relevant chunks and generate answer
        retrieved_chunks = retrieve_context(user_query, chunks, top_k=3)
        answer = generate_answer(user_query, retrieved_chunks)
        
        # Display assistant answer
        with st.chat_message("assistant"):
            st.markdown(answer)
            
            # Display retrieved sources and chunks in expandable section
            if retrieved_chunks:
                with st.expander("📚 View Retrieved Sources & Context Chunks"):
                    for idx, chunk in enumerate(retrieved_chunks, start=1):
                        st.markdown(f"**Chunk #{idx} — Source:** `{chunk.get('source')}` | **Chunk ID:** `{chunk.get('chunk_id')}`")
                        st.code(chunk.get("text", ""), language="text")


def main() -> None:
    """Main application execution pipeline."""
    setup_page_configuration()
    documents, chunks = load_and_process_documents()
    render_sidebar(len(documents), len(chunks))
    render_main_content(chunks)


if __name__ == "__main__":
    main()
