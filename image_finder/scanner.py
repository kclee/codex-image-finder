"""Read-only library discovery and rebuildable thumbnail generation."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True, slots=True)
class ScanProgress:
    discovered: int
    current_path: str


@dataclass(frozen=True, slots=True)
class ScanSummary:
    scan_id: str
    discovered: int
    hashed: int
    unchanged: int
    thumbnails_created: int
    errors: int
    missing_marked: int


ProgressCallback = Callable[[ScanProgress], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_metadata(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _create_thumbnail(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.jpg")
    with Image.open(source) as image:
        image.seek(0)
        converted = ImageOps.exif_transpose(image).convert("RGB")
        converted.thumbnail((480, 270), Image.Resampling.LANCZOS)
        converted.save(temporary, format="JPEG", quality=84, optimize=True)
    temporary.replace(destination)


def scan_library(
    connection: sqlite3.Connection,
    project_root: Path,
    library_root: Path,
    progress: ProgressCallback | None = None,
) -> ScanSummary:
    """Discover images without modifying anything below ``library_root``."""

    root = library_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)

    started_at = _now()
    scan_id = str(uuid.uuid4())
    connection.execute(
        "INSERT OR IGNORE INTO libraries(root_path, created_at) VALUES (?, ?)",
        (str(root), started_at),
    )
    library_id = connection.execute(
        "SELECT id FROM libraries WHERE root_path = ?", (str(root),)
    ).fetchone()["id"]
    connection.execute(
        "INSERT INTO scan_runs(id, library_id, started_at, status) VALUES (?, ?, ?, 'running')",
        (scan_id, library_id, started_at),
    )
    connection.commit()

    counters = {
        "discovered": 0,
        "hashed": 0,
        "unchanged": 0,
        "thumbnails": 0,
        "errors": 0,
    }
    walk_errors: list[OSError] = []

    def on_walk_error(error: OSError) -> None:
        walk_errors.append(error)

    try:
        for directory, _subdirectories, filenames in os.walk(
            root, followlinks=False, onerror=on_walk_error
        ):
            for filename in filenames:
                path = Path(directory) / filename
                if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                counters["discovered"] += 1
                relative_path = path.relative_to(root).as_posix()
                if progress:
                    progress(ScanProgress(counters["discovered"], relative_path))
                try:
                    stat = path.stat()
                    existing = connection.execute(
                        """
                        SELECT fl.image_id, fl.source_size, fl.source_modified_ns,
                               i.content_sha256
                        FROM file_locations fl
                        JOIN images i ON i.id = fl.image_id
                        WHERE fl.library_id = ? AND fl.relative_path = ?
                        """,
                        (library_id, relative_path),
                    ).fetchone()

                    if (
                        existing
                        and existing["source_size"] == stat.st_size
                        and existing["source_modified_ns"] == stat.st_mtime_ns
                    ):
                        image_id = existing["image_id"]
                        content_hash = existing["content_sha256"]
                        counters["unchanged"] += 1
                    else:
                        content_hash = _sha256(path)
                        counters["hashed"] += 1
                        width, height = _image_metadata(path)
                        image_id = str(
                            uuid.uuid5(uuid.NAMESPACE_URL, f"sha256:{content_hash}")
                        )
                        connection.execute(
                            """
                            INSERT INTO images(
                                id, content_sha256, byte_size, width, height, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            ON CONFLICT(content_sha256) DO UPDATE SET
                                byte_size=excluded.byte_size,
                                width=excluded.width,
                                height=excluded.height
                            """,
                            (
                                image_id,
                                content_hash,
                                stat.st_size,
                                width,
                                height,
                                started_at,
                            ),
                        )
                        image_id = connection.execute(
                            "SELECT id FROM images WHERE content_sha256 = ?",
                            (content_hash,),
                        ).fetchone()["id"]

                    connection.execute(
                        """
                        INSERT INTO file_locations(
                            library_id, image_id, relative_path, observed_full_path,
                            first_seen_at, last_seen_at, source_size,
                            source_modified_ns, last_scan_token, is_present
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                        ON CONFLICT(library_id, relative_path) DO UPDATE SET
                            image_id=excluded.image_id,
                            observed_full_path=excluded.observed_full_path,
                            last_seen_at=excluded.last_seen_at,
                            source_size=excluded.source_size,
                            source_modified_ns=excluded.source_modified_ns,
                            last_scan_token=excluded.last_scan_token,
                            is_present=1
                        """,
                        (
                            library_id,
                            image_id,
                            relative_path,
                            str(path),
                            started_at,
                            started_at,
                            stat.st_size,
                            stat.st_mtime_ns,
                            scan_id,
                        ),
                    )

                    thumbnail_relative = (
                        Path("data")
                        / "thumbnails"
                        / content_hash[:2]
                        / f"{content_hash}.jpg"
                    )
                    thumbnail_path = project_root / thumbnail_relative
                    if not thumbnail_path.is_file():
                        _create_thumbnail(path, thumbnail_path)
                        counters["thumbnails"] += 1
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO derived_files(
                            image_id, run_id, kind, relative_path, content_sha256, created_at
                        ) VALUES (?, NULL, 'thumbnail', ?, NULL, ?)
                        """,
                        (image_id, thumbnail_relative.as_posix(), started_at),
                    )
                    if counters["discovered"] % 100 == 0:
                        connection.commit()
                except (OSError, ValueError, Image.UnidentifiedImageError):
                    counters["errors"] += 1

        counters["errors"] += len(walk_errors)
        missing_marked = 0
        if not walk_errors:
            cursor = connection.execute(
                """
                UPDATE file_locations
                SET is_present = 0
                WHERE library_id = ?
                  AND is_present = 1
                  AND (last_scan_token IS NULL OR last_scan_token <> ?)
                """,
                (library_id, scan_id),
            )
            missing_marked = cursor.rowcount

        completed_at = _now()
        connection.execute(
            "UPDATE libraries SET last_scan_completed_at = ? WHERE id = ?",
            (completed_at, library_id),
        )
        connection.execute(
            """
            UPDATE scan_runs SET completed_at=?, status='complete',
                discovered_count=?, hashed_count=?, unchanged_count=?,
                thumbnail_count=?, error_count=?
            WHERE id=?
            """,
            (
                completed_at,
                counters["discovered"],
                counters["hashed"],
                counters["unchanged"],
                counters["thumbnails"],
                counters["errors"],
                scan_id,
            ),
        )
        connection.commit()
        return ScanSummary(
            scan_id=scan_id,
            discovered=counters["discovered"],
            hashed=counters["hashed"],
            unchanged=counters["unchanged"],
            thumbnails_created=counters["thumbnails"],
            errors=counters["errors"],
            missing_marked=missing_marked,
        )
    except Exception:
        connection.execute(
            "UPDATE scan_runs SET completed_at=?, status='failed', error_count=? WHERE id=?",
            (_now(), counters["errors"] + 1, scan_id),
        )
        connection.commit()
        raise
