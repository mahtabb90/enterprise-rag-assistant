"""
Document Ingestion Module.

Loads text and markdown documents from a specified local directory.
"""

from pathlib import Path


def ingest_documents(source_path: str = "data/sample_docs") -> list[dict]:
    """Scans a local directory and loads .txt and .md files into a list of document objects.

    Args:
        source_path (str): Relative or absolute path to the document directory.

    Returns:
        list[dict]: A list of dictionaries containing:
            - "source": The filename of the document (e.g., "security_guidelines.txt")
            - "text": The full string content of the file
    """
    directory = Path(source_path)

    # Return an empty list if the specified directory does not exist
    if not directory.exists() or not directory.is_dir():
        print(f"[Warning] Ingestion path '{source_path}' does not exist or is not a directory.")
        return []

    documents = []

    # Iterate through all files in the directory
    for file_path in sorted(directory.iterdir()):
        # Only process .txt and .md files
        if file_path.is_file() and file_path.suffix.lower() in [".txt", ".md"]:
            try:
                content = file_path.read_text(encoding="utf-8")
                documents.append({
                    "source": file_path.name,
                    "text": content
                })
            except Exception as e:
                print(f"[Error] Failed to read '{file_path.name}': {e}")

    return documents
