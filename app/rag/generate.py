"""
Answer Generation Module.

Synthesizes local grounded responses using retrieved document context.
"""


def generate_answer(query: str, context: list[dict]) -> str:
    """Generates a grounded answer based on retrieved document chunks.

    Args:
        query (str): The user query string.
        context (list[dict]): List of retrieved chunk dictionaries.

    Returns:
        str: A grounded answer string formatted for the local prototype.
    """
    if not context:
        return (
            "I could not find an answer to your question in the available local documents. "
            "Please try rephrasing your search terms or asking about password requirements, "
            "annual leave policies, or API rate limits."
        )

    # Extract distinct document sources referenced in context
    sources = sorted(list({chunk.get("source", "Unknown") for chunk in context}))
    source_list_str = ", ".join(sources)

    # Build response snippet summaries from context chunks
    snippets = []
    for idx, chunk in enumerate(context, start=1):
        source = chunk.get("source", "Unknown")
        text_preview = chunk.get("text", "").strip()
        snippets.append(f"**[{idx}] {source} (Chunk #{chunk.get('chunk_id', '?')})**:\n\"{text_preview}\"")

    snippets_text = "\n\n".join(snippets)

    response = (
        f"**Answer (Local Rule-Based Prototype)**:\n\n"
        f"Based on the retrieved context from **{source_list_str}**, here is the relevant information matching your query:\n\n"
        f"{snippets_text}"
    )

    return response
