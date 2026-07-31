import hashlib
import re
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
        """Read a document and split it on semantic boundaries.

        The configured size remains a character budget for compatibility, but
        chunks now prefer paragraph and sentence boundaries instead of cutting
        through arbitrary words or table rows.
        """
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

        # Keep offsets by splitting the original text rather than rebuilding
        # it from stripped paragraphs.
        units = []
        for match in re.finditer(r"\S[\s\S]*?(?:(?:\n\s*){2,}|$)", text):
            start, end = match.span()
            unit = text[start:end].strip()
            if unit:
                leading = len(text[start:end]) - len(text[start:end].lstrip())
                units.append((start + leading, start + leading + len(unit), unit))

        if not units:
            units = [(0, len(text), text.strip())]

        chunks: List[TextChunk] = []
        current: List[tuple] = []
        current_len = 0

        def flush() -> None:
            nonlocal current, current_len
            if not current:
                return
            start = current[0][0]
            end = current[-1][1]
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = hashlib.sha256(
                    f"{doc_id}_{start}_{chunk_text[:80]}".encode("utf-8")
                ).hexdigest()[:16]
                chunks.append(TextChunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    text=chunk_text,
                    start_idx=start,
                    end_idx=end,
                ))
            current = []
            current_len = 0

        for start, end, unit in units:
            # Very long paragraphs are split at sentence boundaries.
            sentence_units = [(start, end, unit)]
            if len(unit) > self.chunk_size:
                sentence_units = []
                cursor = start
                for sentence in re.finditer(r"[^.!?]+[.!?]?(?:\s+|$)", unit, re.S):
                    sentence_text = sentence.group(0).strip()
                    if sentence_text:
                        local_start = unit.find(sentence_text, max(0, cursor - start))
                        absolute_start = start + (local_start if local_start >= 0 else 0)
                        sentence_units.append((absolute_start, absolute_start + len(sentence_text), sentence_text))
                        cursor = absolute_start + len(sentence_text)
                if not sentence_units:
                    sentence_units = [(start, end, unit)]

            for item in sentence_units:
                item_len = len(item[2])
                if current and current_len + item_len + 1 > self.chunk_size:
                    previous = current[-1:]
                    flush()
                    # Preserve a small semantic overlap rather than a raw
                    # character slice.
                    current = previous
                    current_len = len(previous[0][2]) if previous else 0
                current.append(item)
                current_len += item_len + 1

        flush()

        return chunks
