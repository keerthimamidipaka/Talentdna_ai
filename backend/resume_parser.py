"""Resume Intelligence Engine (Module 1): extract raw text from PDF/DOCX."""

import io
import fitz  # PyMuPDF
import docx


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    f = io.BytesIO(file_bytes)
    document = docx.Document(f)
    return "\n".join(p.text for p in document.paragraphs)


def extract_resume_text(filename: str, file_bytes: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif lower.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif lower.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def guess_candidate_name(resume_text: str) -> str:
    """Naive heuristic: first non-empty line that looks like a name (Module 1)."""
    for line in resume_text.splitlines():
        line = line.strip()
        if 2 <= len(line.split()) <= 4 and len(line) < 40 and not any(c.isdigit() for c in line):
            if "@" not in line and "http" not in line.lower():
                return line
    return "Candidate"
