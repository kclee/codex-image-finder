"""Catalog importing and search services, independent of any UI framework."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .database import connect
from .domain import SearchResult
from .scanner import ProgressCallback, ScanSummary, scan_library


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Catalog:
    def __init__(self, database_path: Path, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.database_path = database_path.resolve()
        self.connection = connect(self.database_path)

    def close(self) -> None:
        self.connection.close()

    def import_ocr_trial(self, json_path: Path) -> int:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        items = payload.get("items", [])
        if not items:
            return 0

        source_root = self._infer_source_root(items[0])
        timestamp = _now()
        library_id = self._upsert_library(source_root, timestamp)

        configuration = payload.get("configuration", {})
        engine_label = configuration.get("engine", "PaddleOCR")
        run_id = hashlib.sha256(
            json.dumps(configuration, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]
        self.connection.execute(
            """
            INSERT OR IGNORE INTO analysis_runs(
                id, analysis_type, engine_name, engine_version, model_name,
                model_version, pipeline_version, parameters_json, started_at, completed_at
            ) VALUES (?, 'ocr', 'PaddleOCR', '3.7.0', ?, 'PP-OCRv5',
                      'trial-v1', ?, ?, ?)
            """,
            (
                run_id,
                engine_label,
                json.dumps(configuration, ensure_ascii=False, sort_keys=True),
                payload.get("created_at", timestamp),
                payload.get("created_at", timestamp),
            ),
        )

        imported = 0
        for item in items:
            absolute_path = Path(item["absolute_path"])
            if not absolute_path.is_file():
                continue
            content_hash = _file_sha256(absolute_path)
            image_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"sha256:{content_hash}"))
            self.connection.execute(
                """
                INSERT INTO images(id, content_sha256, byte_size, width, height, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(content_sha256) DO UPDATE SET
                    byte_size=excluded.byte_size,
                    width=excluded.width,
                    height=excluded.height
                """,
                (
                    image_id,
                    content_hash,
                    item.get("bytes", absolute_path.stat().st_size),
                    item.get("width", 0),
                    item.get("height", 0),
                    timestamp,
                ),
            )
            stored_id = self.connection.execute(
                "SELECT id FROM images WHERE content_sha256 = ?", (content_hash,)
            ).fetchone()["id"]
            self.connection.execute(
                """
                INSERT INTO file_locations(
                    library_id, image_id, relative_path, observed_full_path,
                    first_seen_at, last_seen_at, is_present
                ) VALUES (?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(library_id, relative_path) DO UPDATE SET
                    image_id=excluded.image_id,
                    observed_full_path=excluded.observed_full_path,
                    last_seen_at=excluded.last_seen_at,
                    is_present=1
                """,
                (
                    library_id,
                    stored_id,
                    item["relative_path"],
                    str(absolute_path),
                    timestamp,
                    timestamp,
                ),
            )
            self.connection.execute(
                """
                INSERT INTO analysis_results(
                    run_id, image_id, all_text, subtitle_text, confidence,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, image_id) DO UPDATE SET
                    all_text=excluded.all_text,
                    subtitle_text=excluded.subtitle_text,
                    confidence=excluded.confidence,
                    payload_json=excluded.payload_json
                """,
                (
                    run_id,
                    stored_id,
                    item.get("all_text", ""),
                    item.get("subtitle_text", ""),
                    item.get("average_confidence"),
                    json.dumps(item, ensure_ascii=False, sort_keys=True),
                    timestamp,
                ),
            )
            thumbnail = self.project_root / item["thumbnail"]
            self.connection.execute(
                """
                INSERT OR IGNORE INTO derived_files(
                    image_id, run_id, kind, relative_path, created_at
                ) VALUES (?, ?, 'thumbnail', ?, ?)
                """,
                (stored_id, run_id, str(thumbnail.relative_to(self.project_root)), timestamp),
            )
            imported += 1

        self.connection.commit()
        return imported

    def groups(self) -> list[str]:
        rows = self.connection.execute(
            "SELECT DISTINCT relative_path FROM file_locations WHERE is_present = 1"
        )
        groups = {
            self._group_from_path(row["relative_path"])
            for row in rows
        }
        return sorted(groups, key=str.casefold)

    def library_roots(self) -> list[Path]:
        rows = self.connection.execute(
            "SELECT root_path FROM libraries ORDER BY created_at"
        ).fetchall()
        return [Path(row["root_path"]) for row in rows]

    def scan_library(
        self, root: Path, progress: ProgressCallback | None = None
    ) -> ScanSummary:
        return scan_library(self.connection, self.project_root, root, progress)

    def latest_analysis_batch_image_ids(self, run_id: str) -> set[str]:
        latest = self.connection.execute(
            "SELECT queued_at FROM analysis_jobs WHERE run_id=? "
            "ORDER BY queued_at DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if latest is None:
            return set()
        rows = self.connection.execute(
            "SELECT image_id FROM analysis_jobs WHERE run_id=? AND queued_at=?",
            (run_id, latest["queued_at"]),
        )
        return {row["image_id"] for row in rows}

    def search(self, query: str = "", group: str | None = None) -> list[SearchResult]:
        clauses = ["fl.is_present = 1"]
        arguments: list[object] = []
        if query.strip():
            clauses.append(
                "(instr(lower(ar.all_text), lower(?)) > 0 "
                "OR instr(lower(ar.subtitle_text), lower(?)) > 0 "
                "OR instr(lower(fl.relative_path), lower(?)) > 0)"
            )
            arguments.extend([query.strip()] * 3)
        if group:
            if group == "(archive root)":
                clauses.append("instr(fl.relative_path, '/') = 0 AND instr(fl.relative_path, '\\') = 0")
            else:
                clauses.append("(fl.relative_path LIKE ? OR fl.relative_path LIKE ?)")
                arguments.extend([f"{group}/%", f"{group}\\%"])

        rows = self.connection.execute(
            f"""
            SELECT i.id AS image_id, fl.relative_path, fl.observed_full_path,
                   i.width, i.height, ar.all_text, ar.subtitle_text, ar.confidence,
                   ar.run_id AS analysis_run_id, ar.created_at AS analysis_created_at,
                   analysis_run.engine_name AS analysis_engine,
                   analysis_run.model_name AS analysis_model,
                   analysis_run.pipeline_version AS analysis_pipeline,
                   df.relative_path AS thumbnail_path
            FROM images i
            JOIN file_locations fl ON fl.image_id = i.id
            LEFT JOIN analysis_results ar ON ar.id = (
                SELECT ar2.id
                FROM analysis_results ar2
                JOIN analysis_runs run2 ON run2.id = ar2.run_id
                WHERE ar2.image_id = i.id
                ORDER BY COALESCE(run2.completed_at, run2.started_at) DESC, ar2.id DESC
                LIMIT 1
            )
            LEFT JOIN analysis_runs analysis_run ON analysis_run.id = ar.run_id
            LEFT JOIN derived_files df ON df.id = (
                SELECT df2.id
                FROM derived_files df2
                WHERE df2.image_id = i.id AND df2.kind = 'thumbnail'
                ORDER BY (df2.run_id IS NULL) DESC, df2.id DESC
                LIMIT 1
            )
            WHERE {' AND '.join(clauses)}
            ORDER BY fl.relative_path COLLATE NOCASE
            """,
            arguments,
        ).fetchall()

        results: list[SearchResult] = []
        for row in rows:
            relative_path = row["relative_path"]
            thumbnail = row["thumbnail_path"] or ""
            results.append(
                SearchResult(
                    image_id=row["image_id"],
                    relative_path=relative_path,
                    absolute_path=Path(row["observed_full_path"]),
                    thumbnail_path=self.project_root / thumbnail,
                    group_name=self._group_from_path(relative_path),
                    width=row["width"],
                    height=row["height"],
                    all_text=row["all_text"] or "",
                    subtitle_text=row["subtitle_text"] or "",
                    confidence=row["confidence"],
                    analysis_run_id=row["analysis_run_id"],
                    analysis_engine=row["analysis_engine"],
                    analysis_model=row["analysis_model"],
                    analysis_pipeline=row["analysis_pipeline"],
                    analysis_created_at=row["analysis_created_at"],
                )
            )
        return results

    def _upsert_library(self, root: Path, timestamp: str) -> int:
        self.connection.execute(
            "INSERT OR IGNORE INTO libraries(root_path, created_at) VALUES (?, ?)",
            (str(root), timestamp),
        )
        return self.connection.execute(
            "SELECT id FROM libraries WHERE root_path = ?", (str(root),)
        ).fetchone()["id"]

    @staticmethod
    def _infer_source_root(item: dict[str, object]) -> Path:
        absolute_path = Path(str(item["absolute_path"]))
        relative_path = Path(str(item["relative_path"]))
        root = absolute_path
        for _ in relative_path.parts:
            root = root.parent
        return root

    @staticmethod
    def _group_from_path(relative_path: str) -> str:
        normalized = relative_path.replace("\\", "/")
        return normalized.split("/", 1)[0] if "/" in normalized else "(archive root)"
