"""
TF-IDF Vectorization Module.

Provides low-level TF-IDF lexical vectorization primitives used during
document indexing and query-time retrieval.

Terminology note
----------------
TF-IDF (Term Frequency-Inverse Document Frequency) produces *lexical vectors*
based on word occurrence statistics.  These are NOT semantic embeddings.
The name ``embeddings.py`` is intentionally reserved for a future stage
when a real embedding model (e.g. Vertex AI text-embedding-004) is introduced.

Shared feature space
--------------------
A TfidfVectorizer must first be *fitted* on the document corpus.  Fitting
defines the vocabulary (which words exist) and the IDF weights (how rare each
word is across the corpus).

Once fitted, BOTH the document corpus and any incoming query must be
*transformed* using that same fitted vectorizer.  This guarantees that document
vectors and query vectors live in the same feature space, which is a prerequisite
for cosine similarity to be meaningful.

The vectorizer must NEVER be re-fitted on a query.

Note on future cloud embedding providers
-----------------------------------------
A pretrained model such as Vertex AI text-embedding-004 does not need to be
fitted on this local corpus — it comes pre-trained.  However, the same principle
applies: documents and queries must both be encoded by the SAME model to share a
compatible vector space.  The vectorize_corpus / vectorize_query separation here
models that same responsibility boundary.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix


def build_vectorizer(corpus: list[str]) -> TfidfVectorizer:
    """Fit a TfidfVectorizer on the document corpus and return it.

    This function defines the vocabulary and IDF weights from the corpus.
    It should be called ONCE during document indexing and never at query time.

    TF-IDF intuition
    ----------------
    - Term Frequency (TF): how often a word appears in a single document.
    - Inverse Document Frequency (IDF): how rare the word is across all
      documents.  Common words like "the" get low IDF; rare domain-specific
      words get high IDF.
    - The TF-IDF score of a word in a document = TF x IDF.

    Args:
        corpus (list[str]): List of text strings -- one entry per document chunk.

    Returns:
        TfidfVectorizer: A fitted vectorizer ready to transform text into
            TF-IDF sparse vectors.
    """
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    return vectorizer


def vectorize_corpus(vectorizer: TfidfVectorizer, corpus: list[str]) -> spmatrix:
    """Transform the document corpus into a TF-IDF matrix.

    Uses the already-fitted vectorizer so every document vector is expressed
    in the same vocabulary / feature space defined during indexing.

    Called ONCE during indexing.  The resulting matrix is stored and reused
    for every subsequent query without rebuilding.

    Args:
        vectorizer (TfidfVectorizer): A fitted TfidfVectorizer (from
            ``build_vectorizer``).
        corpus (list[str]): The same list of text strings used for fitting.

    Returns:
        spmatrix: A sparse matrix of shape (n_chunks, n_features) where each
            row is the TF-IDF vector for one document chunk.
    """
    # transform() uses the fitted vocabulary without modifying the vectorizer
    return vectorizer.transform(corpus)


def vectorize_query(vectorizer: TfidfVectorizer, query: str) -> spmatrix:
    """Transform a single query string into a TF-IDF vector.

    Uses the same fitted vectorizer as the corpus so the query vector lives
    in the same feature space as all document vectors.

    Called at QUERY TIME only -- once per user question.

    Words in the query that were not seen during fitting (out-of-vocabulary)
    are silently ignored.  This is expected TF-IDF behaviour.

    Args:
        vectorizer (TfidfVectorizer): The same fitted TfidfVectorizer used for
            the corpus (from ``build_vectorizer``).
        query (str): The user''s question or search string.

    Returns:
        spmatrix: A sparse matrix of shape (1, n_features) representing the
            query as a TF-IDF vector in the corpus feature space.
    """
    # transform() on a single string requires a list wrapper
    return vectorizer.transform([query])
