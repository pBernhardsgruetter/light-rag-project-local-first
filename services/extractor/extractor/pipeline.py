import argparse
import asyncio
import json
from pathlib import Path
import yaml

try:
    import transformers.utils.import_utils as import_utils
    import_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from .chunking import SemanticChunker
from .ner import EntityExtractor
from .re import RelationExtractor
from .resolution import EntityResolver
from .ingestion import KuzuIngestor

class CheckpointManager:
    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self.processed = self._load()

    def _load(self) -> set:
        if self.checkpoint_file.exists():
            try:
                data = json.loads(self.checkpoint_file.read_text(encoding="utf-8"))
                return set(data.get("processed", []))
            except Exception:
                return set()
        return set()

    def mark_processed(self, file_path: Path):
        self.processed.add(str(file_path.resolve()))
        self.checkpoint_file.write_text(
            json.dumps({"processed": list(self.processed)}, indent=2),
            encoding="utf-8"
        )

    def is_processed(self, file_path: Path) -> bool:
        return str(file_path.resolve()) in self.processed

async def run_pipeline(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    input_dir = Path(cfg["input_dir"])
    input_dir.mkdir(parents=True, exist_ok=True)
    kuzu_path = Path(cfg["kuzu_path"])
    checkpoint_file = Path(cfg["checkpoint_file"])

    checkpoint = CheckpointManager(checkpoint_file)
    chunker = SemanticChunker(
        chunk_size=cfg.get("chunk_size", 1000),
        chunk_overlap=cfg.get("chunk_overlap", 200)
    )
    ner = EntityExtractor(
        model_name=cfg["gliner_model"],
        entity_types=cfg["entity_types"],
        batch_size=cfg.get("batch_size", 32)
    )
    re_extractor = RelationExtractor(
        model_name=cfg["rebel_model"],
        batch_size=cfg.get("batch_size", 32)
    )
    resolver = EntityResolver(
        embedding_model=cfg["embedding_model"],
        similarity_threshold=cfg.get("resolution", {}).get("similarity_threshold", 0.85),
        openrouter_key=cfg.get("openrouter_key", ""),
        openrouter_model=cfg.get("openrouter_model", "")
    )
    supported_extensions = {".txt", ".md", ".jsonl", ".pdf"}
    files = [p for p in input_dir.glob("**/*") if p.is_file() and p.suffix.lower() in supported_extensions]

    if not files:
        print(f"No input files found in {input_dir}")
        return

    for doc_path in files:
        if checkpoint.is_processed(doc_path):
            print(f"Skipping already processed file: {doc_path}")
            continue

        print(f"Processing document: {doc_path}")
        chunks = chunker.chunk_document(doc_path)
        if not chunks:
            checkpoint.mark_processed(doc_path)
            continue

        entities = ner.extract_entities(chunks)
        relations = re_extractor.extract_relations(chunks)
        resolved = await resolver.resolve_entities(entities)

        ingestor = KuzuIngestor(db_path=kuzu_path, embedding_model=cfg["embedding_model"])
        ingestor.ingest_document(doc_path, chunks, resolved, relations)
        ingestor.close()
        checkpoint.mark_processed(doc_path)
        print(f"Successfully ingested {doc_path}: {len(chunks)} chunks, {len(resolved)} entities, {len(relations)} relations.")

def main():
    parser = argparse.ArgumentParser(description="GraphRAG Extraction Pipeline")
    parser.add_argument("--config", type=str, default="config/extractor.yaml", help="Path to config file")
    args = parser.parse_args()
    asyncio.run(run_pipeline(Path(args.config)))

if __name__ == "__main__":
    main()
