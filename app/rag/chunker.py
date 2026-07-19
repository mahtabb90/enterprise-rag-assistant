"""
Text chunking module for breaking documents into smaller text segments.
"""

def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    """Splits a body of text into smaller overlapping chunks.

    Args:
        text (str): The input text document to be split into chunks.
        chunk_size (int): The maximum character length of each chunk. Defaults to 100.
        overlap (int): The number of overlapping characters between consecutive chunks. Defaults to 20.

    Returns:
        list[str]: A list of text chunk strings.

    Example:
        >>> chunk_text("Hello world example text", chunk_size=10, overlap=2)
        ['Hello worl', 'rld exampl', 'ple text']
    """
    # Handle empty or whitespace-only input
    if not text or not text.strip():
        return []

    # If the text is shorter than or equal to the chunk size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]

    # Ensure valid overlap step calculation to prevent infinite loops
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be strictly greater than overlap.")

    chunks = []
    start = 0
    text_len = len(text)

    # Slide a window across the text to produce overlapping chunks
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Advance the starting position by (chunk_size - overlap)
        start += step

    return chunks
