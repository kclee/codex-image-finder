"""Persistent user-authored review decisions, separate from derived analysis data."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


REVIEW_DECISIONS = ("accepted", "needs_correction", "not_relevant")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewStore:
    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path)
        self._closed = False
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS image_reviews (
                image_id TEXT NOT NULL,
                analysis_run_id TEXT NOT NULL,
                decision TEXT NOT NULL CHECK (
                    decision IN ('accepted', 'needs_correction', 'not_relevant')
                ),
                updated_at TEXT NOT NULL,
                PRIMARY KEY(image_id, analysis_run_id)
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        if not self._closed:
            self.connection.close()
            self._closed = True

    def decision(self, image_id: str, analysis_run_id: str) -> str | None:
        row = self.connection.execute(
            "SELECT decision FROM image_reviews "
            "WHERE image_id=? AND analysis_run_id=?",
            (image_id, analysis_run_id),
        ).fetchone()
        return row["decision"] if row else None

    def decisions_for_run(self, analysis_run_id: str) -> dict[str, str]:
        rows = self.connection.execute(
            "SELECT image_id, decision FROM image_reviews WHERE analysis_run_id=?",
            (analysis_run_id,),
        )
        return {row["image_id"]: row["decision"] for row in rows}

    def set_decision(
        self, image_id: str, analysis_run_id: str, decision: str
    ) -> None:
        if decision not in REVIEW_DECISIONS:
            raise ValueError(f"Unsupported review decision: {decision}")
        self.connection.execute(
            """
            INSERT INTO image_reviews(image_id, analysis_run_id, decision, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(image_id, analysis_run_id) DO UPDATE SET
                decision=excluded.decision,
                updated_at=excluded.updated_at
            """,
            (image_id, analysis_run_id, decision, _now()),
        )
        self.connection.commit()

    def clear_decision(self, image_id: str, analysis_run_id: str) -> None:
        self.connection.execute(
            "DELETE FROM image_reviews WHERE image_id=? AND analysis_run_id=?",
            (image_id, analysis_run_id),
        )
        self.connection.commit()
