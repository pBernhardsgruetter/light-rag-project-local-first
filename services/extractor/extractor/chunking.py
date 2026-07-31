import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class TextChunk:
    id: str
    doc_id: str
    text: str
    start_idx: int
    end_idx: int

class SemanticChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc_path: Path) -> List[TextChunk]:
        """Reads document and splits text into overlapping chunks."""
        if doc_path.suffix.lower() == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(doc_path))
                text_pages = []
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_pages.append(extracted)
                text = "\n\n".join(text_pages)
            except Exception as e:
                print(f"Error reading PDF {doc_path}: {e}")
                text = ""
        else:
            text = doc_path.read_text(encoding="utf-8", errors="ignore")
        doc_id = hashlib.sha256(str(doc_path.resolve()).encode("utf-8")).hexdigest()[:16]

        if not text.strip():
            return []

        chunks: List[TextChunk] = []
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = self.chunk_size

        for i in range(0, len(text), step):
            chunk_text = text[i:i + self.chunk_size]
            if not chunk_text.strip():
                continue
            chunk_id = hashlib.sha256(f"{doc_id}_{i}_{chunk_text[:30]}".encode("utf-8")).hexdigest()[:16]
            chunks.append(TextChunk(
                id=chunk_id,
                doc_id=doc_id,
                text=chunk_text,
                start_idx=i,
                end_idx=i + len(chunk_text)
            ))

        return chunks
