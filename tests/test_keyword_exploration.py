from __future__ import annotations

import unittest

from image_finder.keyword_exploration import extract_keyword_candidates


class KeywordExplorationTests(unittest.TestCase):
    def test_candidates_are_deterministic_normalized_and_noise_filtered(self) -> None:
        texts = [
            "合成角色非常開心",
            "合成角色真的開心",
            "合成角色今天開心",
            "合成角色一起開心",
            "測試人物向你道歉",
            "測試人物正式道歉",
            "測試人物再次道歉",
            "測試人物需要道歉",
            "哈哈哈哈 這個 那個",
        ]
        first = extract_keyword_candidates(
            texts, minimum_documents=4, maximum_document_ratio=1.0, limit=10
        )
        second = extract_keyword_candidates(
            reversed(texts), minimum_documents=4, maximum_document_ratio=1.0, limit=10
        )
        self.assertEqual(first, second)
        terms = {candidate.text for candidate in first}
        self.assertTrue(any("開心" in term for term in terms))
        self.assertTrue(any("道歉" in term for term in terms))
        self.assertNotIn("哈哈", terms)
        self.assertNotIn("這個", terms)


if __name__ == "__main__":
    unittest.main()
