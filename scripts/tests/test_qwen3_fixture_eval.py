import unittest

from scripts.qwen3_fixture_eval import (
    find_missing_expected_filenames,
    parse_expectations,
    query_hit_count,
    query_passes,
)


class ParseExpectationsTests(unittest.TestCase):
    def test_parse_expectations_supports_dash_and_colon(self) -> None:
        text = """
Clusters by text query:

"cat" - cat_1.jpg, cat_2.jpg
"woman in office": woman_office.jpg
"how are you feeling today?": baileys.png, guiness.png
"""
        got = parse_expectations(text)
        self.assertEqual(
            got,
            [
                ("cat", ["cat_1.jpg", "cat_2.jpg"]),
                ("woman in office", ["woman_office.jpg"]),
                (
                    "how are you feeling today?",
                    ["baileys.png", "guiness.png"],
                ),
            ],
        )


class QueryEvaluationTests(unittest.TestCase):
    def test_query_hit_count_uses_top_k(self) -> None:
        scored = [
            ("a.jpg", 0.9),
            ("b.jpg", 0.8),
            ("c.jpg", 0.7),
            ("d.jpg", 0.6),
        ]
        expected = ["a.jpg", "d.jpg"]
        self.assertEqual(query_hit_count(scored, expected), 1)

    def test_query_passes_relaxed_allows_one_miss_for_larger_cluster(self) -> None:
        self.assertTrue(query_passes(hit_count=3, expected_count=4, strict=False))
        self.assertFalse(query_passes(hit_count=2, expected_count=4, strict=False))

    def test_query_passes_strict_requires_all_hits(self) -> None:
        self.assertTrue(query_passes(hit_count=2, expected_count=2, strict=True))
        self.assertFalse(query_passes(hit_count=1, expected_count=2, strict=True))


class MissingExpectedNamesTests(unittest.TestCase):
    def test_find_missing_expected_filenames(self) -> None:
        images = ["guiness.png", "stream.png"]
        expectations = [
            ("q", ["guinness.png", "stream.png"]),
        ]
        self.assertEqual(
            find_missing_expected_filenames(images, expectations), {"guinness.png"}
        )


if __name__ == "__main__":
    unittest.main()
