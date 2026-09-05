from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from image_finder.analysis_queue import AnalysisQueue, AnalysisSpec
from image_finder.analysis_specs import MOBILE_SUBTITLE_SPEC
from image_finder.catalog import Catalog
from image_finder.database import connect
from image_finder.ocr_engine import _extract_lines


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
                    payload={"source": "unit-test"},
                )
                self.assertEqual(queue.counts(run_id)["succeeded"], 1)
                self.assertEqual(queue.enqueue_missing(run_id), 0)
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


if __name__ == "__main__":
    unittest.main()
