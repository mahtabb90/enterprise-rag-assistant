"""
Enterprise RAG Assistant - Main Streamlit Entry Point

Run locally with:
    streamlit run app/main.py
"""

import streamlit as st
import config


def setup_page_configuration() -> None:
    """Configures the Streamlit browser window title and page layout."""
    st.set_page_config(
        page_title=config.APP_TITLE,
        page_icon="🤖",
        layout="wide"
    )


def render_sidebar() -> None:
    """Renders project status and overview information in the sidebar."""
    st.sidebar.title("📊 Project Status")
    
    st.sidebar.info(
        "**Phase 1: Local Foundation**\n\n"
        "• Core architecture & module structure established\n"
        "• Local text chunker implementation\n"
        "• Unit test suite configured"
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
    
    st.sidebar.caption("Status: Day 1 Skeleton Ready")


def render_main_content() -> None:
    """Renders the main page title, explanation, and interactive Q&A interface."""
    st.title(config.APP_TITLE)
    
    st.markdown(
        """
        Welcome to the **Enterprise RAG Assistant**.
        
        ### What this app will become
        This application is being built step-by-step as a demonstration of a 
        **production-grade Retrieval-Augmented Generation (RAG) system**.
        
        Future capabilities will include:
        1. **Document Ingestion**: Parsing structured and unstructured enterprise documents.
        2. **Smart Chunking**: Splitting text into semantic segments for embedding indexing.
        3. **Vector Retrieval**: Searching relevant context from a high-performance vector store.
        4. **Grounded Generation**: Synthesizing accurate answers using advanced LLMs with source citations.
        """
    )
    
    st.divider()
    
    st.subheader("💬 Ask the Assistant")
    st.caption("Enter a question below to test the interface interaction.")
    
    # User query input field
    user_query = st.chat_input("Ask a question about your enterprise documents...")
    
    if user_query:
        # Display the user's prompt
        with st.chat_message("user"):
            st.write(user_query)
            
        # Display a mock assistant answer for Day 1
        with st.chat_message("assistant"):
            st.markdown(
                f"**Mock Answer:**\n\n"
                f"Thank you for asking: *\"{user_query}\"*\n\n"
                "*(This is a mock response. Live embedding retrieval and LLM answer "
                "generation will be integrated in upcoming development stages.)*"
            )


def main() -> None:
    """Main execution workflow for the Streamlit application."""
    setup_page_configuration()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
