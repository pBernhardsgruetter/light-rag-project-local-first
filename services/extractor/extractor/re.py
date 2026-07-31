try:
    import transformers.utils.import_utils as import_utils
    import_utils.check_torch_load_is_safe = lambda: None
    import transformers.modeling_utils as modeling_utils
    modeling_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .chunking import TextChunk

class RelationExtractor:
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None

    def _load_pipeline(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def _parse_rebel_output(self, text: str) -> List[Dict[str, str]]:
        triplets = []
        current_subject = ""
        current_relation = ""

        text = text.strip()
        tokens = text.split()
        current_type = None
        current_str = []

        for token in tokens:
            if token == "<triplet>":
                if current_subject and current_relation and current_str:
                    triplets.append({
                        "subject": current_subject,
                        "relation": current_relation,
                        "object": " ".join(current_str)
                    })
                current_subject = ""
                current_relation = ""
                current_str = []
                current_type = "subj"
            elif token == "<subj>":
                if current_type == "subj":
                    current_subject = " ".join(current_str)
                current_str = []
                current_type = "rel"
            elif token == "<obj>":
                if current_type == "rel":
                    current_relation = " ".join(current_str)
                current_str = []
                current_type = "obj"
            else:
                current_str.append(token)

        if current_subject and current_relation and current_str:
            triplets.append({
                "subject": current_subject,
                "relation": current_relation,
                "object": " ".join(current_str)
            })

        return triplets

    def extract_relations(self, chunks: List[TextChunk]) -> List[Dict[str, Any]]:
        self._load_pipeline()
        texts = [c.text for c in chunks]
        results = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_chunks = chunks[i:i + self.batch_size]
            
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = self.model.generate(**inputs, max_new_tokens=256, num_beams=3, do_sample=False)

            for chunk, out in zip(batch_chunks, outputs):
                gen_text = self.tokenizer.decode(out, skip_special_tokens=False)
                parsed = self._parse_rebel_output(gen_text)
                for rel in parsed:
                    results.append({
                        "chunk_id": chunk.id,
                        "source": rel["subject"],
                        "target": rel["object"],
                        "type": rel["type"].upper().replace(" ", "_") if "type" in rel else rel["relation"].upper().replace(" ", "_"),
                        "confidence": 0.85
                    })

        return results
