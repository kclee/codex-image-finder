from __future__ import annotations

import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

from PIL import Image

from image_finder.analysis_queue import AnalysisQueue, AnalysisSpec
from image_finder.analysis_specs import MOBILE_SUBTITLE_SPEC
from image_finder.catalog import Catalog
from image_finder.database import connect
from image_finder.ocr_engine import PaddleSubtitleOcr, _extract_lines
from image_finder.review import ocr_review_reason
from image_finder.review_store import ReviewStore
from image_finder.text_search import query_variants, to_traditional


class DatabaseSchemaTests(unittest.TestCase):
    def test_schema_has_identity_and_versioning_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = connect(Path(directory) / "test.sqlite3")
            try:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                self.assertTrue(
                    {
                        "images",
                        "file_locations",
                        "analysis_runs",
                        "analysis_results",
                        "derived_files",
                    }.issubset(tables)
                )
                image_columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(images)")
                }
                self.assertIn("content_sha256", image_columns)
                run_columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(analysis_runs)")
                }
                self.assertTrue(
                    {"engine_version", "model_version", "pipeline_version"}.issubset(run_columns)
                )
            finally:
                connection.close()


class IncrementalScannerTests(unittest.TestCase):
    def test_unchanged_rescan_reuses_identity_without_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            library = root / "library"
            project.mkdir()
            library.mkdir()
            Image.new("RGB", (80, 45), "purple").save(library / "unchanged.png")

            catalog = Catalog(project / "data" / "catalog.sqlite3", project)
            try:
                first = catalog.scan_library(library)
                second = catalog.scan_library(library)
                self.assertEqual(first.hashed, 1)
                self.assertEqual(second.discovered, 1)
                self.assertEqual(second.hashed, 0)
                self.assertEqual(second.unchanged, 1)
                self.assertEqual(second.thumbnails_created, 0)
            finally:
                catalog.close()

    def test_rename_preserves_image_identity_and_location_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            library = root / "library"
            project.mkdir()
            library.mkdir()
            original = library / "before.png"
            Image.new("RGB", (80, 45), "navy").save(original)

            catalog = Catalog(project / "data" / "catalog.sqlite3", project)
            try:
                first = catalog.scan_library(library)
                self.assertEqual(first.discovered, 1)
                self.assertEqual(first.hashed, 1)
                original.rename(library / "after.png")
                second = catalog.scan_library(library)

                self.assertEqual(second.discovered, 1)
                self.assertEqual(
                    catalog.connection.execute("SELECT COUNT(*) FROM images").fetchone()[0],
                    1,
                )
                locations = catalog.connection.execute(
                    "SELECT relative_path, is_present FROM file_locations ORDER BY relative_path"
                ).fetchall()
                self.assertEqual(
                    [(row["relative_path"], row["is_present"]) for row in locations],
                    [("after.png", 1), ("before.png", 0)],
                )
            finally:
                catalog.close()


class FullCatalogSearchTests(unittest.TestCase):
    def test_search_returns_each_current_location_once(self) -> None:
        project = Path(__file__).resolve().parent.parent
        database = project / "data" / "image-finder.sqlite3"
        if not database.exists():
            self.skipTest("full disposable catalog has not been built")
        catalog = Catalog(database, project)
        try:
            present = catalog.connection.execute(
                "SELECT COUNT(*) FROM file_locations WHERE is_present = 1"
            ).fetchone()[0]
            self.assertEqual(len(catalog.search()), present)
        finally:
            catalog.close()


class AnalysisQueueTests(unittest.TestCase):
    def test_queue_is_versioned_persistent_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            library = root / "library"
            project.mkdir()
            library.mkdir()
            Image.new("RGB", (80, 45), "teal").save(library / "one.png")
            Image.new("RGB", (80, 45), "orange").save(library / "two.png")

            catalog = Catalog(project / "data" / "catalog.sqlite3", project)
            try:
                catalog.scan_library(library)
                queue = AnalysisQueue(catalog.connection)
                spec = AnalysisSpec(
                    analysis_type="ocr",
                    engine_name="fake-ocr",
                    engine_version="1.0",
                    model_name="fake-mobile",
                    model_version="1",
                    pipeline_version="lower-crop-v1",
                    parameters={"crop_start": 0.45},
                )
                run_id = queue.ensure_run(spec)
                self.assertEqual(queue.enqueue_missing(run_id), 2)
                first = queue.claim_next(run_id)
                self.assertIsNotNone(first)
                self.assertEqual(queue.counts(run_id)["running"], 1)

                self.assertEqual(queue.recover_interrupted(run_id), 1)
                resumed = queue.claim_next(run_id)
                self.assertEqual(resumed.image_id, first.image_id)
                queue.complete(
                    resumed,
                    all_text="測試",
                    subtitle_text="測試",
                    confidence=0.9,
                    payload={"source": "unit-test", "elapsed_seconds": 1.25},
                )
                self.assertEqual(queue.counts(run_id)["succeeded"], 1)
                self.assertEqual(queue.enqueue_missing(run_id), 0)
                history = catalog.analysis_history(resumed.image_id)
                self.assertEqual(len(history), 1)
                self.assertEqual(history[0].pipeline_version, "lower-crop-v1")
                self.assertEqual(history[0].subtitle_text, "測試")
                timing = catalog.analysis_timing(run_id)
                self.assertIsNotNone(timing)
                self.assertEqual(timing.sample_count, 1)
                self.assertEqual(timing.median_seconds, 1.25)
                batches = catalog.analysis_batch_summaries(run_id)
                self.assertEqual(len(batches), 1)
                self.assertEqual(batches[0].total, 2)
                self.assertEqual(batches[0].succeeded, 1)
                self.assertEqual(batches[0].pending, 1)
                self.assertEqual(batches[0].inference_seconds, 1.25)
            finally:
                catalog.close()

    def test_bounded_enqueue_never_prepares_more_than_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            library = root / "library"
            project.mkdir()
            library.mkdir()
            for number in range(12):
                Image.new("RGB", (20, 20), (number, 0, 0)).save(
                    library / f"{number:02}.png"
                )
            catalog = Catalog(project / "data" / "catalog.sqlite3", project)
            try:
                catalog.scan_library(library)
                queue = AnalysisQueue(catalog.connection)
                run_id = queue.ensure_run(MOBILE_SUBTITLE_SPEC)
                self.assertEqual(queue.enqueue_next_missing(run_id, 5), 5)
                self.assertEqual(queue.counts(run_id)["pending"], 5)
                self.assertEqual(queue.enqueue_next_missing(run_id, 5), 0)
                self.assertEqual(queue.counts(run_id)["pending"], 5)
                self.assertEqual(len(catalog.latest_analysis_batch_image_ids(run_id)), 0)
                ordered = list(reversed([
                    row[0]
                    for row in catalog.connection.execute(
                        "SELECT id FROM images ORDER BY id"
                    )
                ]))
                prepared = queue.prepare_ordered_batch(run_id, ordered, 5)
                self.assertEqual(len(prepared), 5)
                self.assertEqual(queue.counts(run_id)["pending"], 5)
                preview, total = queue.preview_ordered_batch(run_id, ordered, 5)
                self.assertEqual(preview, prepared)
                self.assertEqual(total, 12)
                all_remaining, total = queue.preview_ordered_batch(
                    run_id, ordered, len(ordered)
                )
                self.assertEqual(total, 12)
                self.assertEqual(len(all_remaining), 12)
                self.assertEqual(set(all_remaining), set(ordered))
                claimed = [queue.claim_next(run_id, prepared).image_id for _ in range(5)]
                self.assertEqual(claimed, prepared)
            finally:
                catalog.close()

    def test_analysis_version_changes_when_pipeline_changes(self) -> None:
        changed = AnalysisSpec(
            analysis_type=MOBILE_SUBTITLE_SPEC.analysis_type,
            engine_name=MOBILE_SUBTITLE_SPEC.engine_name,
            engine_version=MOBILE_SUBTITLE_SPEC.engine_version,
            model_name=MOBILE_SUBTITLE_SPEC.model_name,
            model_version=MOBILE_SUBTITLE_SPEC.model_version,
            pipeline_version="different-pipeline",
            parameters=MOBILE_SUBTITLE_SPEC.parameters,
        )
        self.assertNotEqual(changed.run_id, MOBILE_SUBTITLE_SPEC.run_id)


class OcrExtractionTests(unittest.TestCase):
    def test_extract_lines_filters_confidence_and_upper_region(self) -> None:
        payload = {
            "rec_texts": ["上方", "好想吃冰淇淋哦", "low confidence"],
            "rec_scores": [0.9, 0.95, 0.2],
            "rec_boxes": [[0, 5, 30, 15], [0, 60, 100, 80], [0, 70, 50, 90]],
        }
        lines = _extract_lines(payload, lower_only=True, image_height=100)
        self.assertEqual([line["text"] for line in lines], ["好想吃冰淇淋哦"])

    def test_full_frame_fallback_uses_pillow_decoded_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gif-content-with-jpg-extension.jpg"
            Image.new("RGB", (80, 45), "purple").save(path, format="GIF")

            class FakeOcr:
                def __init__(self) -> None:
                    self.inputs: list[object] = []

                def predict(self, value: object) -> list[SimpleNamespace]:
                    self.inputs.append(value)
                    text = "English" if len(self.inputs) == 1 else "測試字幕"
                    return [
                        SimpleNamespace(
                            json={
                                "res": {
                                    "rec_texts": [text],
                                    "rec_scores": [0.95],
                                    "rec_boxes": [[0, 20, 70, 40]],
                                }
                            }
                        )
                    ]

            adapter = PaddleSubtitleOcr.__new__(PaddleSubtitleOcr)
            adapter.ocr = FakeOcr()
            output = adapter.analyze(path)

            self.assertEqual(output.subtitle_text, "測試字幕")
            self.assertEqual(output.payload["pass"], "full-frame-fallback")
            self.assertEqual(len(adapter.ocr.inputs), 2)
            self.assertTrue(all(not isinstance(value, str) for value in adapter.ocr.inputs))


class ChineseSearchTests(unittest.TestCase):
    def test_traditional_query_expands_to_simplified(self) -> None:
        variants = query_variants("你是在教訓我嗎")
        self.assertIn("你是在教训我吗", variants)

    def test_copy_conversion_produces_traditional_chinese(self) -> None:
        self.assertEqual(to_traditional("你是在教训我吗"), "你是在教訓我嗎")


class ReviewRuleTests(unittest.TestCase):
    def test_review_reason_distinguishes_missing_low_and_good_results(self) -> None:
        self.assertEqual(
            ocr_review_reason("", 0.98),
            "No Chinese/Japanese subtitle was selected",
        )
        self.assertEqual(
            ocr_review_reason("友人A", 0.42),
            "Low OCR confidence (42.0%)",
        )
        self.assertIsNone(ocr_review_reason("好想吃冰淇淋哦", 0.99))

    def test_review_decisions_persist_separately_and_can_be_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "user-data" / "review-state.sqlite3"
            store = ReviewStore(database)
            store.set_decision("stable-image", "ocr-version", "not_relevant")
            store.close()

            reopened = ReviewStore(database)
            try:
                self.assertEqual(
                    reopened.decision("stable-image", "ocr-version"),
                    "not_relevant",
                )
                self.assertEqual(
                    reopened.decisions_for_run("ocr-version"),
                    {"stable-image": "not_relevant"},
                )
                reopened.clear_decision("stable-image", "ocr-version")
                self.assertIsNone(
                    reopened.decision("stable-image", "ocr-version")
                )
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()
