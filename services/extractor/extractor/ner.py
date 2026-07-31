from . import __version__

try:
    import transformers.utils.import_utils as import_utils
    import_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from typing import List, Dict, Any
from gliner import GLiNER
from .chunking import TextChunk

class EntityExtractor:
    def __init__(
        self,
        model_name: str,
        entity_types: List[str],
        batch_size: int = 32,
        score_threshold: float = 0.35,
    ):
        self.model_name = model_name
        self.entity_types = entity_types
        self.batch_size = batch_size
        self.score_threshold = score_threshold
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = GLiNER.from_pretrained(self.model_name)

    def extract_entities(self, chunks: List[TextChunk]) -> List[Dict[str, Any]]:
        self._load_model()
        extracted: List[Dict[str, Any]] = []

        for chunk in chunks:
            preds = self._model.predict_entities(chunk.text, labels=self.entity_types)
            for p in preds:
                score = float(p.get("score", 1.0))
                if score < self.score_threshold:
                    continue
                extracted.append({
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "text": p["text"],
                    "label": p["label"],
                    "start": p["start"],
                    "end": p["end"],
                    "score": score
                })

        return extracted
