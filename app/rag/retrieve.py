"""
Keyword-Based Retrieval Module.

Performs keyword overlap matching between user queries and document chunks.
"""

import string


def _normalize_tokens(text: str) -> set[str]:
    """Helper function to lower-case text, strip basic punctuation, and extract unique word tokens.

    Args:
        text (str): Input text string.

    Returns:
        set[str]: Set of cleaned, lower-cased word tokens.
    """
    # Remove basic punctuation characters
    cleaned_text = text.lower().translate(str.maketrans("", "", string.punctuation))
    # Split by whitespace into word tokens
    return set(cleaned_text.split())


def retrieve_context(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """Retrieves top_k document chunks relevant to a query based on keyword overlap.

    Args:
        query (str): The search query entered by the user.
        chunks (list[dict]): List of chunk dictionaries containing 'source', 'chunk_id', and 'text'.
        top_k (int): Maximum number of top matching chunks to return. Defaults to 3.

    Returns:
        list[dict]: List of retrieved top matching chunk objects, sorted by relevance score.
            Chunks with zero keyword overlap are excluded.
    """
    query_tokens = _normalize_tokens(query)

    # Return empty list if query has no valid word tokens
    if not query_tokens:
        return []

    scored_chunks = []

    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        chunk_tokens = _normalize_tokens(chunk_text)

        # Count common word tokens between query and chunk
        score = len(query_tokens.intersection(chunk_tokens))

        # Ignore chunks with zero keyword overlap
        if score > 0:
            scored_chunks.append((score, chunk))

    # Sort chunks by overlap score in descending order
    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    # Extract and return top_k chunk objects
    return [chunk for score, chunk in scored_chunks[:top_k]]
