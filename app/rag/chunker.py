"""
Text Chunking Module.

Splits documents and raw text strings into smaller overlapping segments for indexing,
ensuring clean word boundary alignment.
"""


def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    """Splits a body of text into smaller overlapping chunks, attempting to align boundaries on word spaces.

    Args:
        text (str): The input text document to be split into chunks.
        chunk_size (int): The maximum character length of each chunk. Defaults to 100.
        overlap (int): The number of overlapping characters between consecutive chunks. Defaults to 20.

    Returns:
        list[str]: A list of cleaned text chunk strings.
    """
    if not text or not text.strip():
        return []

    clean_input = text.strip()

    if len(clean_input) <= chunk_size:
        return [clean_input]

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be strictly greater than overlap.")

    chunks = []
    start = 0
    text_len = len(clean_input)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Snap to word boundary if not at the end of the string
        if end < text_len:
            last_space = clean_input.rfind(" ", start, end)
            last_newline = clean_input.rfind("\n", start, end)
            boundary = max(last_space, last_newline)
            if boundary > start:
                end = boundary

        chunk = clean_input[start:end].strip()
        if chunk and (not chunks or chunks[-1] != chunk):
            chunks.append(chunk)

        # Advance starting index
        start += step

        # Prevent infinite loops if step doesn't move past current end
        if start <= (end - chunk_size) and end < text_len:
            start = end

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

        # Split text into clean word-aligned segments
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
