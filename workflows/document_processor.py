
import logging
import os
from typing import List
try:
    import pypdf
except ImportError:
    pypdf = None
try:
    from docx import Document
except ImportError:
    Document = None

logger = logging.getLogger("document-processor")

def process_file(file_path: str) -> str:
    """Parse PDF, DOCX, or TXT into plain text."""
    if not os.path.exists(file_path):
        return ""
        
    ext = file_path.split(".")[-1].lower()
    try:
        if ext == "pdf" and pypdf:
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext == "docx" and Document:
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    if not text: return []
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks
