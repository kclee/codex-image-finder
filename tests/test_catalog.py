from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from image_finder.catalog import Catalog
from image_finder.database import connect


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


if __name__ == "__main__":
    unittest.main()
