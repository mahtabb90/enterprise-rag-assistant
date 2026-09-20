"""
Retrieval Module.

Provides two retrieval strategies:

1. TF-IDF Vector Retrieval (primary)
   - prepare_tfidf_index()  -- INDEXING phase: fit vectorizer, transform corpus once.
   - retrieve_tfidf()       -- QUERY-TIME phase: transform query only, compute similarity.

2. Keyword Retrieval (fallback)
   - retrieve_context()     -- Normalized keyword overlap matching.

RAG indexing / retrieval distinction
-------------------------------------
prepare_tfidf_index() mirrors the *document indexing* phase of a RAG pipeline:
  build once, store for reuse.
retrieve_tfidf() mirrors the *query-time retrieval* phase:
  fast lookup against the pre-built index.
In a production system this boundary maps to:
  offline index building (embedding vectors into a vector database)
  vs. online query serving (embed query -> nearest-neighbour search).

Result schema
-------------
Both retrieval paths return a list of dicts with a consistent schema:
  {
      "source":           str   -- original filename,
      "chunk_id":         int   -- sequential chunk identifier,
      "text":             str   -- chunk text content,
      "score":            float -- cosine similarity (TF-IDF) or 0 (keyword),
      "retrieval_method": str   -- "tfidf" or "keyword",
  }
"""

import string
from sklearn.metrics.pairwise import cosine_similarity
from app.rag.vectorization import build_vectorizer, vectorize_corpus, vectorize_query


# ---------------------------------------------------------------------------
# Keyword retrieval helpers
# ---------------------------------------------------------------------------

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

    This function is the keyword-based fallback retrieval strategy.  It is called
    automatically by retrieve_tfidf() when TF-IDF produces no results above its
    threshold, or it can be used directly.

    Args:
        query (str): Search query entered by the user.
        chunks (list[dict]): List of chunk dicts with source, chunk_id, and text.
        top_k (int): Maximum number of matching chunks to return. Defaults to 3.
        min_score (int): Minimum required meaningful term match count. Defaults to 1.

    Returns:
        list[dict]: Top matching chunks sorted by relevance score descending.
            Each dict contains: source, chunk_id, text, score (0), retrieval_method ("keyword").
            Returns an empty list if no meaningful query terms match any chunk.
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

    # Build result dicts with consistent schema; keyword results carry score=0
    # because keyword overlap count is not comparable to cosine similarity scores.
    return [
        {
            "source": chunk.get("source"),
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text", ""),
            "score": 0.0,
            "retrieval_method": "keyword",
        }
        for _keyword_score, chunk in scored_chunks[:top_k]
    ]


# ---------------------------------------------------------------------------
# TF-IDF indexing and retrieval
# ---------------------------------------------------------------------------

def prepare_tfidf_index(chunks: list[dict]) -> dict:
    """Build and return a TF-IDF index from document chunks.

    INDEXING PHASE -- call this once at application startup, not at query time.

    Steps:
        1. Extract the text from every chunk to form the corpus.
        2. Fit a TfidfVectorizer on the corpus (defines vocabulary + IDF weights).
        3. Transform the entire corpus into a sparse TF-IDF matrix.
        4. Return all three artefacts bundled as an index dict.

    The returned index is designed to be cached (e.g. with @st.cache_resource)
    and passed into retrieve_tfidf() for every subsequent query.

    Args:
        chunks (list[dict]): List of chunk dicts with source, chunk_id, text.

    Returns:
        dict: {
            "vectorizer":    fitted TfidfVectorizer,
            "corpus_matrix": sparse TF-IDF matrix (n_chunks x n_features),
            "chunks":        original list of chunk dicts (preserved for metadata lookup),
        }
    """
    # Step 1: extract plain text strings -- one per chunk
    corpus = [chunk.get("text", "") for chunk in chunks]

    # Step 2: fit the vectorizer on the corpus.
    # This defines the feature space: vocabulary and IDF weights.
    # Must only happen once on the document corpus, never on a query.
    vectorizer = build_vectorizer(corpus)

    # Step 3: transform every chunk text into a TF-IDF vector.
    # Result is a sparse matrix: rows = chunks, columns = vocabulary terms.
    corpus_matrix = vectorize_corpus(vectorizer, corpus)

    return {
        "vectorizer": vectorizer,
        "corpus_matrix": corpus_matrix,
        "chunks": chunks,
    }


def retrieve_tfidf(
    query: str,
    index: dict,
    top_k: int = 3,
    min_score: float = 0.05,
) -> list[dict]:
    """Retrieve the most relevant chunks for a query using TF-IDF cosine similarity.

    QUERY-TIME PHASE -- call this for every user question.  Only the query is
    transformed here; the corpus matrix is already prepared in the index.

    Steps:
        1. Transform the query into a TF-IDF vector using the already-fitted vectorizer.
        2. Compute cosine similarity between the query vector and every chunk vector.
        3. Filter out chunks below the min_score threshold.
        4. Sort by similarity descending and return the top_k results.
        5. If no chunk passes the threshold, fall back to keyword retrieval.
        6. If keyword retrieval also finds no match, return an empty list.

    Cosine similarity intuition
    ---------------------------
    Cosine similarity measures the angle between two vectors.
      - Score of 1.0 means the query and chunk point in exactly the same direction
        (identical term distribution).
      - Score of 0.0 means the query and chunk share no common weighted terms.
    Unlike raw dot product, cosine similarity is not affected by vector length,
    so longer chunks are not unfairly penalised.

    Args:
        query (str): The user question or search string.
        index (dict): The prepared TF-IDF index from prepare_tfidf_index().
        top_k (int): Maximum number of results to return. Defaults to 3.
        min_score (float): Minimum cosine similarity to include a result (0.0-1.0).
            Defaults to 0.05.  Results below this threshold are considered irrelevant.
            Note: small local corpora naturally produce lower TF-IDF scores because
            many terms appear across multiple documents, reducing IDF weights.

    Returns:
        list[dict]: Ranked list of matching chunk dicts, each containing:
            source, chunk_id, text, score (float), retrieval_method ("tfidf").
            Falls back to keyword retrieval results (retrieval_method="keyword")
            if TF-IDF finds nothing above the threshold.
            Returns [] if neither method finds a meaningful match.
    """
    vectorizer = index["vectorizer"]
    corpus_matrix = index["corpus_matrix"]
    chunks = index["chunks"]

    # Step 1: transform only the query -- the vectorizer is already fitted.
    # The query vector lives in the same feature space as the corpus matrix.
    query_vector = vectorize_query(vectorizer, query)

    # Step 2: compute cosine similarity between the query and every chunk.
    # cosine_similarity returns a 2D array of shape (1, n_chunks).
    # Flatten to a 1D array indexed by chunk position.
    similarity_scores = cosine_similarity(query_vector, corpus_matrix).flatten()

    # Step 3: collect chunks that pass the relevance threshold
    scored_results = []
    for idx, score in enumerate(similarity_scores):
        if score >= min_score:
            chunk = chunks[idx]
            scored_results.append({
                "source": chunk.get("source"),
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk.get("text", ""),
                "score": float(round(score, 4)),
                "retrieval_method": "tfidf",
            })

    # Step 4: sort by similarity score descending
    scored_results.sort(key=lambda r: r["score"], reverse=True)
    tfidf_results = scored_results[:top_k]

    # Step 5: fall back to keyword retrieval if TF-IDF found nothing above threshold.
    # The keyword fallback still applies its own meaningful-match filter,
    # so completely off-topic queries will also return [] from keyword retrieval.
    if not tfidf_results:
        return retrieve_context(query, chunks, top_k=top_k)

    return tfidf_results
