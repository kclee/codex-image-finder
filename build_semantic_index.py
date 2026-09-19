"""Build or validate the complete local MiniLM subtitle index for Image Finder."""

from __future__ import annotations

import argparse
from pathlib import Path

from image_finder.app_search import (
    SEMANTIC_INDEX_FILENAME,
    build_application_semantic_index,
    estimated_semantic_build_seconds,
)
from image_finder.semantic_search import application_spec, semantic_index_status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Force a complete rebuild")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    catalog = project / "data" / "image-finder.sqlite3"
    index = project / "data" / SEMANTIC_INDEX_FILENAME
    status = semantic_index_status(catalog, index, application_spec())
    if status.ready and not args.rebuild:
        print(f"Meaning index is ready: {status.indexed_count:,} subtitles")
        return 0
    estimate = estimated_semantic_build_seconds(status.eligible_count)
    print(
        f"Building {status.eligible_count:,} local MiniLM subtitle embeddings "
        f"(estimated {estimate / 60:.1f} minutes from the measured trial)…"
    )
    summary = build_application_semantic_index(catalog, project, rebuild=True)
    print(
        f"Meaning index ready: {summary.sample_size:,} subtitles in "
        f"{summary.total_seconds:.1f} seconds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
