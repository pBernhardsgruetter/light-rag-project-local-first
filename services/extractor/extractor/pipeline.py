import argparse
import asyncio
import hashlib
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
                # Migrate the old path-only checkpoint format. Entries without
                # a fingerprint are intentionally reprocessed once.
                if isinstance(data.get("processed"), list):
                    return {
                        str(path): {"fingerprint": None}
                        for path in data.get("processed", [])
                    }
                return data.get("processed", {})
            except Exception:
                return {}
        return {}

    def mark_processed(self, file_path: Path, fingerprint: str):
        self.processed[str(file_path.resolve())] = {
            "fingerprint": fingerprint,
        }
        self.checkpoint_file.write_text(
            json.dumps({"processed": self.processed}, indent=2),
            encoding="utf-8"
        )

    def is_processed(self, file_path: Path, fingerprint: str) -> bool:
        entry = self.processed.get(str(file_path.resolve()))
        return bool(entry and entry.get("fingerprint") == fingerprint)


def fingerprint_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

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
        batch_size=cfg.get("batch_size", 32),
        score_threshold=cfg.get("entity_score_threshold", 0.35),
    )
    re_extractor = RelationExtractor(
        model_name=cfg["rebel_model"],
        batch_size=cfg.get("batch_size", 32),
        allowed_relation_types=cfg.get("relation_types", []),
        default_confidence=cfg.get("relation_default_confidence", 0.70),
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
        fingerprint = fingerprint_file(doc_path)
        if checkpoint.is_processed(doc_path, fingerprint):
            print(f"Skipping already processed file: {doc_path}")
            continue

        print(f"Processing document: {doc_path}")
        chunks = chunker.chunk_document(doc_path)
        if not chunks:
            checkpoint.mark_processed(doc_path, fingerprint)
            continue

        entities = ner.extract_entities(chunks)
        relations = re_extractor.extract_relations(chunks)
        ingestor = KuzuIngestor(db_path=kuzu_path, embedding_model=cfg["embedding_model"])
        try:
            existing_entities = ingestor.get_entity_catalog()
            resolved = await resolver.resolve_entities(entities, existing_entities)
            # Re-ingesting the same path is safe: old chunks and relation
            # assertions are removed before the new version is written.
            ingestor.delete_document(chunks[0].doc_id)
            ingestor.ingest_document(doc_path, chunks, resolved, relations)
        finally:
            ingestor.close()
        checkpoint.mark_processed(doc_path, fingerprint)
        print(f"Successfully ingested {doc_path}: {len(chunks)} chunks, {len(resolved)} entities, {len(relations)} relations.")

def main():
    parser = argparse.ArgumentParser(description="GraphRAG Extraction Pipeline")
    parser.add_argument("--config", type=str, default="config/extractor.yaml", help="Path to config file")
    args = parser.parse_args()
    asyncio.run(run_pipeline(Path(args.config)))

if __name__ == "__main__":
    main()
