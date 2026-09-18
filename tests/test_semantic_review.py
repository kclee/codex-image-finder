from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote


PROJECT = Path(__file__).resolve().parents[1]


class _ReviewHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[str] = []
        self.review_keys: list[str] = []
        self.context_text = ""
        self._in_context = False

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if tag == "img" and attributes.get("src"):
            self.images.append(attributes["src"] or "")
        if tag == "input" and attributes.get("type") == "radio":
            key = attributes.get("data-review-key")
            if key:
                self.review_keys.append(key)
        if tag == "script" and attributes.get("id") == "review-context":
            self._in_context = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_context:
            self._in_context = False

    def handle_data(self, data: str) -> None:
        if self._in_context:
            self.context_text += data


class SemanticReviewReportTests(unittest.TestCase):
    def test_report_keeps_all_results_and_stable_review_context(self) -> None:
        from run_semantic_model_shootout import _render_html

        performance = {
            "model_snapshot_bytes": 1024,
            "model_load_seconds": 0.01,
            "embedding_seconds": 0.02,
            "embedding_subtitles_per_second": 100.0,
            "warm_query_median_seconds": 0.003,
            "warm_query_p95_seconds": 0.004,
            "sqlite_index_bytes": 2048,
        }
        report = {
            "format_version": 1,
            "generated_at": "2026-01-01T00:00:00Z",
            "sample": {"manifest_sha256": "c" * 64},
            "query_set": {
                "version": "synthetic-v1",
                "queries": [{"query": "合成查詢", "category": "synthetic"}],
            },
            "models": [
                {
                    "key": "model-a",
                    "label": "Synthetic model A",
                    "spec": {
                        "model_name": "local/synthetic-a",
                        "model_revision": "a" * 40,
                        "dimensions": 2,
                    },
                    "performance": performance,
                    "queries": [{"results": [{
                        "image_id": "image-a",
                        "relative_path": "GroupAlpha/a.png",
                        "subtitle_text": "合成字幕甲",
                        "rank": 1,
                        "score": 0.8,
                    }]}],
                },
                {
                    "key": "model-b",
                    "label": "Synthetic model B",
                    "spec": {
                        "model_name": "local/synthetic-b",
                        "model_revision": "b" * 40,
                        "dimensions": 2,
                    },
                    "performance": performance,
                    "queries": [{"results": [{
                        "image_id": "image-b",
                        "relative_path": "GroupBeta/b.png",
                        "subtitle_text": "合成字幕乙",
                        "rank": 1,
                        "score": 0.7,
                    }]}],
                },
            ],
        }
        thumbnails = {
            "image-a": "data/thumbnails/synthetic-a.jpg",
            "image-b": "data/thumbnails/synthetic-b.jpg",
        }
        content_hashes = {"image-a": "1" * 64, "image-b": "2" * 64}
        parser = _ReviewHtmlParser()
        parser.feed(_render_html(report, thumbnails, content_hashes))
        context = json.loads(parser.context_text)

        expected = [
            (
                query["query"],
                model["key"],
                model["spec"]["model_revision"],
                result["image_id"],
                content_hashes[result["image_id"]],
                result["relative_path"],
                result["subtitle_text"],
                result["rank"],
                result["score"],
            )
            for query_index, query in enumerate(report["query_set"]["queries"])
            for model in report["models"]
            for result in model["queries"][query_index]["results"]
        ]
        actual = [
            (
                item["query"],
                item["model_key"],
                item["model_revision"],
                item["image_id"],
                item["image_content_sha256"],
                item["relative_path"],
                item["ocr_subtitle"],
                item["rank"],
                item["similarity_score"],
            )
            for item in context["items"]
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(context["total_result_count"], 2)
        self.assertEqual(context["sample_manifest_sha256"], report["sample"]["manifest_sha256"])
        self.assertEqual(len(parser.review_keys), 6)
        self.assertEqual(len(set(parser.review_keys)), 2)
        self.assertEqual(
            [unquote(source) for source in parser.images], list(thumbnails.values())
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for JavaScript tests")
    def test_javascript_persistence_export_import_and_validation(self) -> None:
        completed = subprocess.run(
            [shutil.which("node") or "node", "tests/semantic_review.test.cjs"],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
