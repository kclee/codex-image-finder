"""Persistent, UI-independent queue for versioned image analysis."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class AnalysisSpec:
    analysis_type: str
    engine_name: str
    engine_version: str
    model_name: str
    model_version: str
    pipeline_version: str
    parameters: dict[str, object] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        canonical = json.dumps(
            {
                "analysis_type": self.analysis_type,
                "engine_name": self.engine_name,
                "engine_version": self.engine_version,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "pipeline_version": self.pipeline_version,
                "parameters": self.parameters,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True, slots=True)
class AnalysisJob:
    job_id: int
    run_id: str
    image_id: str
    source_path: Path
    attempt_count: int


class AnalysisQueue:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def ensure_run(self, spec: AnalysisSpec) -> str:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO analysis_runs(
                id, analysis_type, engine_name, engine_version, model_name,
                model_version, pipeline_version, parameters_json, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                spec.run_id,
                spec.analysis_type,
                spec.engine_name,
                spec.engine_version,
                spec.model_name,
                spec.model_version,
                spec.pipeline_version,
                json.dumps(spec.parameters, ensure_ascii=False, sort_keys=True),
                _now(),
            ),
        )
        self.connection.commit()
        return spec.run_id

    def enqueue_missing(self, run_id: str) -> int:
        before = self.connection.total_changes
        self.connection.execute(
            """
            INSERT OR IGNORE INTO analysis_jobs(
                run_id, image_id, status, queued_at
            )
            SELECT ?, i.id, 'pending', ?
            FROM images i
            WHERE EXISTS (
                SELECT 1 FROM file_locations fl
                WHERE fl.image_id = i.id AND fl.is_present = 1
            )
            AND NOT EXISTS (
                SELECT 1 FROM analysis_results ar
                WHERE ar.run_id = ? AND ar.image_id = i.id
            )
            """,
            (run_id, _now(), run_id),
        )
        inserted = self.connection.total_changes - before
        if inserted:
            self.connection.execute(
                "UPDATE analysis_runs SET completed_at=NULL WHERE id=?", (run_id,)
            )
        self.connection.commit()
        return inserted

    def enqueue_next_missing(self, run_id: str, batch_size: int) -> int:
        if batch_size <= 0:
            return 0
        pending = self.counts(run_id)["pending"]
        available_slots = max(0, batch_size - pending)
        if available_slots == 0:
            return 0
        before = self.connection.total_changes
        self.connection.execute(
            """
            INSERT OR IGNORE INTO analysis_jobs(run_id, image_id, status, queued_at)
            SELECT ?, i.id, 'pending', ?
            FROM images i
            WHERE EXISTS (
                SELECT 1 FROM file_locations fl
                WHERE fl.image_id=i.id AND fl.is_present=1
            )
            AND NOT EXISTS (
                SELECT 1 FROM analysis_results ar
                WHERE ar.run_id=? AND ar.image_id=i.id
            )
            AND NOT EXISTS (
                SELECT 1 FROM analysis_jobs aj
                WHERE aj.run_id=? AND aj.image_id=i.id
            )
            ORDER BY i.created_at, i.id
            LIMIT ?
            """,
            (run_id, _now(), run_id, run_id, available_slots),
        )
        inserted = self.connection.total_changes - before
        if inserted:
            self.connection.execute(
                "UPDATE analysis_runs SET completed_at=NULL WHERE id=?", (run_id,)
            )
        self.connection.commit()
        return inserted

    def enqueue_images(self, run_id: str, image_ids: list[str]) -> int:
        if not image_ids:
            return 0
        before = self.connection.total_changes
        queued_at = _now()
        self.connection.executemany(
            """
            INSERT OR IGNORE INTO analysis_jobs(run_id, image_id, status, queued_at)
            SELECT ?, ?, 'pending', ?
            WHERE EXISTS (
                SELECT 1 FROM file_locations fl
                WHERE fl.image_id=? AND fl.is_present=1
            )
            AND NOT EXISTS (
                SELECT 1 FROM analysis_results ar
                WHERE ar.run_id=? AND ar.image_id=?
            )
            """,
            [
                (run_id, image_id, queued_at, image_id, run_id, image_id)
                for image_id in image_ids
            ],
        )
        inserted = self.connection.total_changes - before
        if inserted:
            self.connection.execute(
                "UPDATE analysis_runs SET completed_at=NULL WHERE id=?", (run_id,)
            )
        self.connection.commit()
        return inserted

    def recover_interrupted(self, run_id: str) -> int:
        cursor = self.connection.execute(
            """
            UPDATE analysis_jobs
            SET status='pending', started_at=NULL,
                last_error='Interrupted before completion; returned to queue'
            WHERE run_id=? AND status='running'
            """,
            (run_id,),
        )
        self.connection.commit()
        return cursor.rowcount

    def claim_next(
        self, run_id: str, allowed_image_ids: list[str] | None = None
    ) -> AnalysisJob | None:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            allowed_clause = ""
            arguments: list[object] = [run_id]
            if allowed_image_ids is not None:
                if not allowed_image_ids:
                    self.connection.commit()
                    return None
                placeholders = ",".join("?" for _ in allowed_image_ids)
                allowed_clause = f" AND aj.image_id IN ({placeholders})"
                arguments.extend(allowed_image_ids)
            row = self.connection.execute(
                f"""
                SELECT aj.id, aj.run_id, aj.image_id, aj.attempt_count,
                       (
                           SELECT fl.observed_full_path
                           FROM file_locations fl
                           WHERE fl.image_id=aj.image_id AND fl.is_present=1
                           ORDER BY fl.last_seen_at DESC, fl.id DESC
                           LIMIT 1
                       ) AS source_path
                FROM analysis_jobs aj
                WHERE aj.run_id=? AND aj.status='pending'
                {allowed_clause}
                ORDER BY aj.priority DESC, aj.id
                LIMIT 1
                """,
                arguments,
            ).fetchone()
            if row is None:
                self.connection.commit()
                return None
            if not row["source_path"]:
                self.connection.execute(
                    "UPDATE analysis_jobs SET status='skipped', completed_at=?, "
                    "last_error='No current source location' WHERE id=?",
                    (_now(), row["id"]),
                )
                self.connection.commit()
                return self.claim_next(run_id, allowed_image_ids)
            self.connection.execute(
                """
                UPDATE analysis_jobs
                SET status='running', started_at=?, completed_at=NULL,
                    attempt_count=attempt_count+1, last_error=NULL
                WHERE id=?
                """,
                (_now(), row["id"]),
            )
            self.connection.commit()
            return AnalysisJob(
                job_id=row["id"],
                run_id=row["run_id"],
                image_id=row["image_id"],
                source_path=Path(row["source_path"]),
                attempt_count=row["attempt_count"] + 1,
            )
        except Exception:
            self.connection.rollback()
            raise

    def complete(
        self,
        job: AnalysisJob,
        *,
        all_text: str,
        subtitle_text: str,
        confidence: float | None,
        payload: dict[str, object],
    ) -> None:
        completed_at = _now()
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
                payload_json=excluded.payload_json,
                created_at=excluded.created_at
            """,
            (
                job.run_id,
                job.image_id,
                all_text,
                subtitle_text,
                confidence,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                completed_at,
            ),
        )
        self.connection.execute(
            "UPDATE analysis_jobs SET status='succeeded', completed_at=?, "
            "last_error=NULL WHERE id=?",
            (completed_at, job.job_id),
        )
        self._complete_run_if_finished(job.run_id, completed_at)
        self.connection.commit()

    def fail(self, job: AnalysisJob, error: str) -> None:
        self.connection.execute(
            "UPDATE analysis_jobs SET status='failed', completed_at=?, last_error=? WHERE id=?",
            (_now(), error[:2000], job.job_id),
        )
        self.connection.commit()

    def retry_failed(self, run_id: str) -> int:
        cursor = self.connection.execute(
            """
            UPDATE analysis_jobs
            SET status='pending', started_at=NULL, completed_at=NULL
            WHERE run_id=? AND status='failed'
            """,
            (run_id,),
        )
        self.connection.commit()
        return cursor.rowcount

    def counts(self, run_id: str) -> dict[str, int]:
        counts = {status: 0 for status in ("pending", "running", "succeeded", "failed", "skipped")}
        rows = self.connection.execute(
            "SELECT status, COUNT(*) AS count FROM analysis_jobs WHERE run_id=? GROUP BY status",
            (run_id,),
        )
        for row in rows:
            counts[row["status"]] = row["count"]
        return counts

    def _complete_run_if_finished(self, run_id: str, completed_at: str) -> None:
        unfinished = self.connection.execute(
            "SELECT COUNT(*) FROM analysis_jobs WHERE run_id=? AND status IN ('pending', 'running')",
            (run_id,),
        ).fetchone()[0]
        if unfinished == 0:
            self.connection.execute(
                "UPDATE analysis_runs SET completed_at=? WHERE id=?",
                (completed_at, run_id),
            )
