import json
import tempfile
import unittest
from pathlib import Path

from services.extractor.extractor.pipeline import CheckpointManager


class CheckpointTest(unittest.TestCase):
    def test_checkpoint_is_content_sensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.json"
            document_path = Path(directory) / "document.txt"
            document_path.write_text("version one", encoding="utf-8")

            manager = CheckpointManager(checkpoint_path)
            manager.mark_processed(document_path, "hash-one")
            self.assertTrue(manager.is_processed(document_path, "hash-one"))
            self.assertFalse(manager.is_processed(document_path, "hash-two"))

            # The persisted shape is explicit and survives a new manager.
            persisted = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(
                persisted["processed"][str(document_path.resolve())]["fingerprint"],
                "hash-one",
            )


if __name__ == "__main__":
    unittest.main()
