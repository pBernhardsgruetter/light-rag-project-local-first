import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "services" / "lightrag"))

import server
from services.extractor.extractor.chunking import TextChunk
from services.extractor.extractor.ingestion import KuzuIngestor


class RetrievalSmokeTest(unittest.TestCase):
    def test_lexical_fallback_returns_ranked_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "db"
            ingestor = KuzuIngestor(db_path)
            chunks = [TextChunk("chunk-1", "doc-1", "Acme uses Battery", 0, 17)]
            entities = [
                {
                    "entity_id": "acme",
                    "canonical_name": "Acme",
                    "type": "Organization",
                    "chunk_id": "chunk-1",
                    "doc_id": "doc-1",
                    "original_text": "Acme",
                },
                {
                    "entity_id": "battery",
                    "canonical_name": "Battery",
                    "type": "Technology",
                    "chunk_id": "chunk-1",
                    "doc_id": "doc-1",
                    "original_text": "Battery",
                },
            ]
            ingestor.ingest_document(
                Path("report.txt"),
                chunks,
                entities,
                [{
                    "chunk_id": "chunk-1",
                    "source": "Acme",
                    "target": "Battery",
                    "type": "USES",
                    "confidence": 0.9,
                }],
            )
            ingestor.close()

            server._snapshot = None
            server._snapshot_signature = None
            server._embedder = None
            server._embedder_failed = True
            result = server.retrieve("What does Acme use?", str(db_path), top_k=2)

            self.assertIn("SOURCE CHUNKS", result["context"])
            self.assertEqual(result["citations"][0]["chunk_id"], "chunk-1")
            self.assertEqual(result["edges"][0]["predicate"], "USES")


if __name__ == "__main__":
    unittest.main()
