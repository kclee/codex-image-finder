from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
