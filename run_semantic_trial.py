"""Build and query the bounded, OCR-subtitle-only semantic trial."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
import time
from pathlib import Path

from image_finder.semantic_search import (
    MAX_TRIAL_IMAGES,
    FastEmbedder,
    SemanticSpec,
    build_trial_index,
    semantic_search,
)


DEFAULT_QUERIES = (
    "傻眼",
    "很尷尬，不知道該說什麼",
    "朋友之間意見不合",
    "awkward reaction",
    "I don't know what to say",
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(
        description="Local semantic search over at most 50 OCR subtitles."
    )
    parser.add_argument("queries", nargs="*", default=list(DEFAULT_QUERIES))
    parser.add_argument("--limit", type=int, default=MAX_TRIAL_IMAGES)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parent
    spec = SemanticSpec()
    model_started = time.perf_counter()
    embedder = FastEmbedder(project / "models" / "fastembed", spec)
    model_load_seconds = time.perf_counter() - model_started
    index_path = project / "data" / "semantic-subtitle-trial.sqlite3"
    summary = build_trial_index(
        project / "data" / "image-finder.sqlite3",
        index_path,
        embedder,
        spec,
        limit=args.limit,
        rebuild=args.rebuild,
    )
    report = {
        "spec": spec.as_dict(),
        "model_load_seconds": model_load_seconds,
        "build": asdict(summary),
        "index_bytes": index_path.stat().st_size,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for query in args.queries:
        started = time.perf_counter()
        results = semantic_search(
            index_path,
            query,
            embedder,
            spec,
            limit=args.top_k,
        )
        elapsed = time.perf_counter() - started
        print(f"\nQUERY {query!r} ({elapsed:.4f}s)")
        for rank, result in enumerate(results, 1):
            print(
                f"  {rank}. {result.score:.4f} | {result.relative_path} | "
                f"{result.subtitle_text}"
            )


if __name__ == "__main__":
    main()
