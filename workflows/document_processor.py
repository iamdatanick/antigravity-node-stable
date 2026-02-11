
import os
import pypdf
from docx import Document

def process_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            return "\n".join([page.extract_text() for page in reader.pages])
    elif ext == ".docx":
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        with open(path, "r") as f:
            return f.read()

def chunk_text(text: str, size: int = 1000) -> list[str]:
    return [text[i:i+size] for i in range(0, len(text), size)]
