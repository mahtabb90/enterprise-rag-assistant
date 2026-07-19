"""
Text Chunking Module.

Splits documents and raw text strings into smaller overlapping segments for indexing.
"""


def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    """Splits a body of text into smaller overlapping chunks.

    Args:
        text (str): The input text document to be split into chunks.
        chunk_size (int): The maximum character length of each chunk. Defaults to 100.
        overlap (int): The number of overlapping characters between consecutive chunks. Defaults to 20.

    Returns:
        list[str]: A list of text chunk strings.
    """
    if not text or not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text]

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be strictly greater than overlap.")

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += step

    return chunks


def chunk_documents(documents: list[dict], chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """Processes a list of documents and splits each into structured chunks with metadata.

    Args:
        documents (list[dict]): List of document dicts with 'source' and 'text'.
        chunk_size (int): Max character length per chunk. Defaults to 500.
        overlap (int): Overlap character count. Defaults to 100.

    Returns:
        list[dict]: List of chunk objects, each containing:
            - "source": Original filename
            - "chunk_id": Sequential integer identifier
            - "text": Chunk text content
    """
    all_chunks = []
    chunk_counter = 1

    for doc in documents:
        source_name = doc.get("source", "unknown")
        raw_text = doc.get("text", "")

        # Split text into string segments using chunk_text
        text_segments = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)

        # Attach metadata to each segment
        for segment in text_segments:
            all_chunks.append({
                "source": source_name,
                "chunk_id": chunk_counter,
                "text": segment
            })
            chunk_counter += 1

    return all_chunks
