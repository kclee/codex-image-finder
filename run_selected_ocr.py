"""Explicitly process a small selected set through the persistent OCR queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from image_finder.analysis_queue import AnalysisQueue
from image_finder.analysis_specs import MOBILE_SUBTITLE_SPEC
from image_finder.catalog import Catalog
from image_finder.ocr_engine import PaddleSubtitleOcr


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OCR for explicitly selected paths")
    parser.add_argument(
        "--pause-after",
        type=int,
        help="Stop cleanly after this many completed attempts, leaving the rest pending",
    )
    parser.add_argument("relative_paths", nargs="+")
    args = parser.parse_args()
    if args.pause_after is not None and args.pause_after < 1:
        parser.error("--pause-after must be at least 1")

    project_root = Path(__file__).resolve().parent
    catalog = Catalog(project_root / "data" / "image-finder.sqlite3", project_root)
    try:
        placeholders = ",".join("?" for _ in args.relative_paths)
        rows = catalog.connection.execute(
            f"""
            SELECT DISTINCT image_id, relative_path
            FROM file_locations
            WHERE is_present=1 AND relative_path IN ({placeholders})
            """,
            args.relative_paths,
        ).fetchall()
        found_paths = {row["relative_path"] for row in rows}
        missing = sorted(set(args.relative_paths) - found_paths)
        if missing:
            raise SystemExit(f"Paths not found in catalog: {missing}")

        ids_by_path = {row["relative_path"]: row["image_id"] for row in rows}
        selected_ids = list(
            dict.fromkeys(ids_by_path[path] for path in args.relative_paths)
        )
        queue = AnalysisQueue(catalog.connection)
        run_id = queue.ensure_run(MOBILE_SUBTITLE_SPEC)
        queue.recover_interrupted(run_id)
        queued = queue.enqueue_images(run_id, selected_ids)
        print(f"run_id={run_id} queued={queued}")
        eligible, _total = queue.preview_ordered_batch(
            run_id, selected_ids, len(selected_ids)
        )
        if not eligible:
            print(queue.counts(run_id))
            return 0

        engine = PaddleSubtitleOcr(project_root / "models")
        processed = 0
        while job := queue.claim_next(run_id, selected_ids):
            try:
                output = engine.analyze(job.source_path)
                queue.complete(
                    job,
                    all_text=output.all_text,
                    subtitle_text=output.subtitle_text,
                    confidence=output.confidence,
                    payload=output.payload,
                )
            except Exception as error:
                queue.fail(job, f"{type(error).__name__}: {error}")
                print(f"failed={job.source_path.name} error={error}")
                return 2
            processed += 1
            print(
                f"completed={job.source_path.name} "
                f"subtitle={json.dumps(output.subtitle_text, ensure_ascii=True)} "
                f"seconds={output.payload['elapsed_seconds']}"
            )
            if args.pause_after is not None and processed >= args.pause_after:
                print(f"paused_after={processed}")
                break
        print(queue.counts(run_id))
        return 0
    finally:
        catalog.close()


if __name__ == "__main__":
    raise SystemExit(main())
