"""SQLite persistence with stable image identity and versioned analysis."""

from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_VERSION = 1


def connect(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    migrate(connection)
    return connection


def migrate(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS libraries (
            id INTEGER PRIMARY KEY,
            root_path TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            content_sha256 TEXT NOT NULL UNIQUE,
            byte_size INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS file_locations (
            id INTEGER PRIMARY KEY,
            library_id INTEGER NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
            image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            relative_path TEXT NOT NULL,
            observed_full_path TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            is_present INTEGER NOT NULL DEFAULT 1 CHECK (is_present IN (0, 1)),
            UNIQUE(library_id, relative_path)
        );
        CREATE INDEX IF NOT EXISTS file_locations_image_idx
            ON file_locations(image_id);

        CREATE TABLE IF NOT EXISTS analysis_runs (
            id TEXT PRIMARY KEY,
            analysis_type TEXT NOT NULL,
            engine_name TEXT NOT NULL,
            engine_version TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            pipeline_version TEXT NOT NULL,
            parameters_json TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
            image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            all_text TEXT NOT NULL DEFAULT '',
            subtitle_text TEXT NOT NULL DEFAULT '',
            confidence REAL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(run_id, image_id)
        );
        CREATE INDEX IF NOT EXISTS analysis_results_image_idx
            ON analysis_results(image_id);

        CREATE TABLE IF NOT EXISTS derived_files (
            id INTEGER PRIMARY KEY,
            image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            run_id TEXT REFERENCES analysis_runs(id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            content_sha256 TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(image_id, kind, relative_path)
        );
        """
    )
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(SCHEMA_VERSION),),
    )
    connection.commit()
