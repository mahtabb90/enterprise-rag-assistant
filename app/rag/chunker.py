"""
Text Chunking Module.

Splits documents into overlapping segments that respect natural text
boundaries.  Boundaries are preferred in this order:

    1. Sentence endings  (. ! ? followed by whitespace or end-of-text)
    2. Paragraph breaks  (blank lines / consecutive newlines)
    3. Word boundaries   (spaces or single newlines)
    4. Hard character cut (last resort, only for text with no whitespace)

This preserves complete sentences and avoids mid-word or mid-sentence splits,
which improves embedding quality for semantic retrieval.
"""

import re


def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    """Split a body of text into overlapping chunks at natural boundaries.

    Boundaries are chosen in preference order:
        sentence endings -> paragraph breaks -> word boundaries -> hard cut.

    Args:
        text:       Input text to split.
        chunk_size: Maximum character length of each chunk.  Defaults to 100.
        overlap:    Number of characters of overlap between consecutive chunks.
                    The overlap re-uses the tail of the previous chunk as the
                    start of the next, preserving context across boundaries.
                    Defaults to 20.

    Returns:
        list[str]: Non-empty chunk strings, each stripped of leading/trailing
            whitespace.

    Raises:
        ValueError: If overlap >= chunk_size.
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

        if end < text_len:
            # Search window: the region [start, end] we want to snap within.
            # Prefer boundaries in descending order of quality.
            boundary = _find_boundary(clean_input, start, end)
            if boundary > start:
                end = boundary

        chunk = clean_input[start:end].strip()
        if chunk and (not chunks or chunks[-1] != chunk):
            chunks.append(chunk)

        start += step

        # Guard: if step is smaller than the snapped chunk we actually emitted,
        # advance past where we are to avoid an infinite loop.
        if start < end - step:
            start = max(start, end - overlap)

    return chunks


def _find_boundary(text: str, start: int, end: int) -> int:
    """Return the best split position within [start, end].

    Searches for the rightmost position of each boundary type in priority
    order and returns the first match found, or end if none.

    Priority:
        1. Sentence boundary: . ! ? followed by space / newline
        2. Paragraph break:   \\n\\n or \\r\\n\\r\\n
        3. Newline:           \\n
        4. Word space:        ' '
    """
    window = text[start:end]

    # 1. Rightmost sentence boundary (period/!? followed by whitespace)
    sentence_matches = list(re.finditer(r'[.!?](?=\s)', window))
    if sentence_matches:
        # +1 to include the punctuation mark itself in this chunk
        return start + sentence_matches[-1].start() + 1

    # 2. Rightmost paragraph break (two or more newlines)
    para_matches = list(re.finditer(r'\n{2,}', window))
    if para_matches:
        return start + para_matches[-1].end()

    # 3. Rightmost single newline
    newline_pos = window.rfind('\n')
    if newline_pos > 0:
        return start + newline_pos

    # 4. Rightmost word space
    space_pos = window.rfind(' ')
    if space_pos > 0:
        return start + space_pos

    # 5. Hard cut (no whitespace found — e.g. dense numeric strings in tests)
    return end


def chunk_documents(documents: list[dict], chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """Process a list of documents and split each into structured chunks with metadata.

    Args:
        documents:  List of document dicts, each with 'source' and 'text' keys.
        chunk_size: Maximum character length per chunk.  Defaults to 500.
        overlap:    Overlap character count between consecutive chunks.
                    Defaults to 100.

    Returns:
        list[dict]: Ordered list of chunk dicts, each containing:
            - ``source``:   Original filename.
            - ``chunk_id``: Sequential integer identifier (1-based, global).
            - ``text``:     Chunk text content.
    """
    all_chunks = []
    chunk_counter = 1

    for doc in documents:
        source_name = doc.get("source", "unknown")
        raw_text = doc.get("text", "")

        text_segments = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)

        for segment in text_segments:
            all_chunks.append({
                "source": source_name,
                "chunk_id": chunk_counter,
                "text": segment,
            })
            chunk_counter += 1

    return all_chunks
