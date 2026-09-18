"""Build and compare lexical, semantic, and hybrid subtitle retrieval."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

from image_finder.catalog import Catalog
from image_finder.semantic_search import (
    MAX_EVALUATION_IMAGES,
    FastEmbedder,
    build_trial_index,
    evaluation_spec,
    fuse_rankings,
    indexed_lexical_search,
    representative_subtitle_sample,
    sample_distribution,
    semantic_search,
)


def _semantic_row(rank: int, item: object) -> dict[str, object]:
    return {
        "method": "semantic",
        "rank": rank,
        "score": item.score,
        "relative_path": item.relative_path,
        "subtitle_text": item.subtitle_text,
        "image_id": item.image_id,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--limit", type=int, default=MAX_EVALUATION_IMAGES)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    project = Path(__file__).resolve().parent
    catalog_path = project / "data" / "image-finder.sqlite3"
    index_path = project / "data" / "semantic-subtitle-evaluation-1000.sqlite3"
    query_path = project / "semantic-evaluation-queries.json"
    output_path = project / "results" / "semantic_evaluation_1000.json"
    previous_report = None
    if output_path.is_file():
        previous_report = json.loads(output_path.read_text(encoding="utf-8"))
    query_document = json.loads(query_path.read_text(encoding="utf-8"))
    queries = [item["query"] for item in query_document["queries"]]
    spec = evaluation_spec()

    model_started = time.perf_counter()
    embedder = FastEmbedder(project / "models" / "fastembed", spec)
    model_load_seconds = time.perf_counter() - model_started
    build = build_trial_index(
        catalog_path,
        index_path,
        embedder,
        spec,
        limit=args.limit,
        rebuild=args.rebuild,
    )
    sample = representative_subtitle_sample(catalog_path, args.limit)

    catalog = Catalog(catalog_path, project)
    evaluations: list[dict[str, object]] = []
    semantic_latencies: list[float] = []
    hybrid_latencies: list[float] = []
    try:
        for query in queries:
            exact = catalog.search(query)
            semantic_started = time.perf_counter()
            semantic_all = semantic_search(
                index_path, query, embedder, spec, limit=args.limit
            )
            semantic_seconds = time.perf_counter() - semantic_started
            hybrid_started = time.perf_counter()
            lexical_indexed = indexed_lexical_search(
                index_path, query, spec, limit=args.limit
            )
            hybrid = fuse_rankings(
                semantic_all, lexical_indexed, limit=args.top_k, rrf_k=60
            )
            hybrid_seconds = semantic_seconds + (time.perf_counter() - hybrid_started)
            semantic_latencies.append(semantic_seconds)
            hybrid_latencies.append(hybrid_seconds)
            evaluations.append(
                {
                    "query": query,
                    "exact_partial_total_full_catalog": len(exact),
                    "exact_partial": [
                        {
                            "method": "exact_partial_full_catalog",
                            "rank": rank,
                            "score": None,
                            "relative_path": item.relative_path,
                            "subtitle_text": item.subtitle_text,
                            "image_id": item.image_id,
                        }
                        for rank, item in enumerate(exact[: args.top_k], 1)
                    ],
                    "indexed_lexical_total": len(lexical_indexed),
                    "indexed_lexical": [
                        {
                            "method": "exact_partial_sampled_index",
                            "rank": rank,
                            "score": item.score,
                            "relative_path": item.relative_path,
                            "subtitle_text": item.subtitle_text,
                            "image_id": item.image_id,
                        }
                        for rank, item in enumerate(
                            lexical_indexed[: args.top_k], 1
                        )
                    ],
                    "semantic": [
                        _semantic_row(rank, item)
                        for rank, item in enumerate(semantic_all[: args.top_k], 1)
                    ],
                    "hybrid": [
                        {
                            "method": "hybrid_rrf",
                            "rank": rank,
                            "score": item.hybrid_score,
                            "semantic_score": item.semantic_score,
                            "lexical_score": item.lexical_score,
                            "relative_path": item.relative_path,
                            "subtitle_text": item.subtitle_text,
                            "image_id": item.image_id,
                        }
                        for rank, item in enumerate(hybrid, 1)
                    ],
                    "semantic_seconds": semantic_seconds,
                    "hybrid_seconds": hybrid_seconds,
                }
            )
            semantic_top = semantic_all[0]
            hybrid_top = hybrid[0]
            print(
                f"{query!r}: exact={len(exact)}; "
                f"semantic={semantic_top.relative_path} ({semantic_top.score:.4f}); "
                f"hybrid={hybrid_top.relative_path} ({hybrid_top.hybrid_score:.6f})"
            )
    finally:
        catalog.close()

    raw_vector_bytes = build.sample_size * spec.dimensions * 4
    initial_build = asdict(build)
    if (
        build.reused
        and previous_report
        and previous_report.get("build", {}).get("run_id") == build.run_id
    ):
        initial_build = previous_report.get(
            "initial_build", previous_report["build"]
        )
    model_cache_bytes = sum(
        path.stat().st_size
        for path in (project / "models" / "fastembed").rglob("*")
        if path.is_file()
    )
    report = {
        "format_version": 1,
        "query_set": query_document,
        "spec": spec.as_dict(),
        "hybrid_method": {
            "name": "equal-weight reciprocal-rank fusion",
            "rrf_k": 60,
            "formula": "sum(1 / (60 + rank)) for semantic and indexed lexical ranks",
            "scope": "the same 1000-item sampled index",
        },
        "initial_build": initial_build,
        "build": asdict(build),
        "sample_distribution": sample_distribution(sample),
        "performance": {
            "model_load_seconds": model_load_seconds,
            "embedding_images_per_second": (
                build.sample_size / build.embedding_seconds
                if build.embedding_seconds
                else None
            ),
            "semantic_query_median_seconds": statistics.median(semantic_latencies),
            "semantic_query_mean_seconds": statistics.mean(semantic_latencies),
            "hybrid_query_median_seconds": statistics.median(hybrid_latencies),
            "hybrid_query_mean_seconds": statistics.mean(hybrid_latencies),
            "index_bytes": index_path.stat().st_size,
            "raw_vector_bytes": raw_vector_bytes,
            "model_cache_bytes": model_cache_bytes,
        },
        "evaluations": evaluations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"REPORT={output_path}")
    print(json.dumps(report["performance"], indent=2))


if __name__ == "__main__":
    main()
