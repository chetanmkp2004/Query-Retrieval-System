# utils/document_parser.py
from typing import List

def fetch_document(url: str) -> str:
    """(Member 3 will implement this) Downloads a doc and returns its local path."""
    print("--- Called fetch_document (Not Implemented) ---")
    return "temp_policy.pdf" # Return a dummy path for now

def extract_text_from_pdf(filepath: str) -> str:
    """(Member 3 will implement this) Extracts all text from a PDF."""
    print("--- Called extract_text_from_pdf (Not Implemented) ---")
    return "This is dummy text from a PDF document."

def chunk_text(text: str) -> List[str]:
    """(Member 3 will implement this) Splits text into smaller chunks."""
    print("--- Called chunk_text (Not Implemented) ---")
    return ["chunk 1", "chunk 2", "chunk 3"]
