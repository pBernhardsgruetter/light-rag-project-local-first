import tempfile
import unittest
from pathlib import Path

from services.extractor.extractor.chunking import TextChunk
from services.extractor.extractor.ingestion import KuzuIngestor


class KuzuIngestionSmokeTest(unittest.TestCase):
    def test_relation_assertion_and_document_replacement(self):
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
            relations = [{
                "chunk_id": "chunk-1",
                "source": "Acme",
                "target": "Battery",
                "type": "USES",
                "confidence": 0.9,
            }]

            ingestor.ingest_document(Path("report.txt"), chunks, entities, relations)
            result = ingestor.conn.execute(
                "MATCH (a:RelationAssertion) RETURN a.predicate, a.chunk_id, a.doc_id"
            )
            self.assertTrue(result.has_next())
            self.assertEqual(result.get_next(), ["USES", "chunk-1", "doc-1"])

            ingestor.delete_document("doc-1")
            result = ingestor.conn.execute("MATCH (c:Chunk) RETURN count(c)")
            self.assertEqual(result.get_next()[0], 0)
            result = ingestor.conn.execute(
                "MATCH (a:RelationAssertion) RETURN count(a)"
            )
            self.assertEqual(result.get_next()[0], 0)
            ingestor.close()


if __name__ == "__main__":
    unittest.main()
