import unittest

from services.extractor.extractor.quality import (
    lexical_score,
    normalize_relation_type,
    normalize_text,
    resolve_endpoint,
)


class QualityHelpersTest(unittest.TestCase):
    def test_normalize_text_is_stable(self):
        self.assertEqual(normalize_text("  EV-Adoption  "), "ev adoption")

    def test_lexical_score_rewards_shared_terms(self):
        self.assertGreater(
            lexical_score("EV adoption air quality", "EV adoption improves air quality"),
            lexical_score("EV adoption air quality", "unrelated battery chemistry"),
        )

    def test_relation_alias_is_canonicalized(self):
        self.assertEqual(
            normalize_relation_type("employed by", ["WORKS_FOR"])[0],
            "WORKS_FOR",
        )

    def test_ambiguous_substring_does_not_pick_arbitrarily(self):
        candidates = [("Acme", "one"), ("ACME", "two")]
        self.assertIsNone(resolve_endpoint("Acme", candidates))

    def test_exact_endpoint_wins(self):
        candidates = [("DeepSeek", "one"), ("DeepSeek-V3", "two")]
        self.assertEqual(resolve_endpoint("DeepSeek", candidates), "one")


if __name__ == "__main__":
    unittest.main()
