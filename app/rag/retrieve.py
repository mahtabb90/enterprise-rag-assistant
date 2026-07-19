"""
Keyword-Based Retrieval Module.

Performs normalized keyword relevance matching between user queries and document chunks,
filtering out common stopwords and domain-generic terms.
"""

import string

# Common English stopwords to ignore during keyword scoring
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "get", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your"
}

# Generic domain filler terms to ignore during relevance scoring
GENERIC_WORDS = {
    "company", "information", "document", "documents", "policy", "policies",
    "guideline", "guidelines", "faq", "faqs", "details", "system", "app",
    "assistant", "help"
}


def _tokenize_text(text: str) -> set[str]:
    """Lower-cases text, strips basic punctuation, and returns a set of word tokens."""
    cleaned_text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return set(cleaned_text.split())


def retrieve_context(query: str, chunks: list[dict], top_k: int = 3, min_score: int = 1) -> list[dict]:
    """Retrieves top_k document chunks relevant to a query based on meaningful keyword overlap.

    Args:
        query (str): Search query entered by the user.
        chunks (list[dict]): List of chunk dictionaries containing 'source', 'chunk_id', and 'text'.
        top_k (int): Maximum number of matching chunks to return. Defaults to 3.
        min_score (int): Minimum required meaningful term match count. Defaults to 1.

    Returns:
        list[dict]: Top matching chunk objects sorted by relevance score descending.
            Returns an empty list if no meaningful query terms match.
    """
    raw_tokens = _tokenize_text(query)

    # Filter out stopwords and generic domain terms to keep only meaningful keywords
    meaningful_query_terms = {
        term for term in raw_tokens
        if term not in STOPWORDS and term not in GENERIC_WORDS
    }

    # Return empty list if no meaningful keywords remain after filtering
    if not meaningful_query_terms:
        return []

    scored_chunks = []

    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        chunk_tokens = _tokenize_text(chunk_text)

        # Count common meaningful tokens between query and chunk
        score = len(meaningful_query_terms.intersection(chunk_tokens))

        # Enforce minimum relevance threshold
        if score >= min_score:
            scored_chunks.append((score, chunk))

    # Sort chunks by overlap score descending
    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    # Extract top_k matching chunk objects
    return [chunk for score, chunk in scored_chunks[:top_k]]
