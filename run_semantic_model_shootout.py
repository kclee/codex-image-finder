"""Run the fixed 1,000-subtitle / 16-query local embedding-model shootout."""

from __future__ import annotations

import argparse
import gc
import html
import json
import math
import sqlite3
import statistics
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

import psutil

from image_finder.semantic_search import (
    MAX_EVALUATION_IMAGES,
    FastEmbedder,
    SemanticSpec,
    build_trial_index,
    model_shootout_specs,
    representative_subtitle_sample,
    sample_manifest_hash,
    sample_distribution,
    semantic_search,
)


MODEL_LABELS = {
    "minilm": "MiniLM baseline",
    "e5-small": "Multilingual E5-small",
    "mpnet": "Multilingual MPNet int8",
}


class RssSampler:
    """Low-overhead process working-set sampler; values are observations, not limits."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.start_bytes = self.process.memory_info().rss
        self.peak_bytes = self.start_bytes
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.05):
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    def __enter__(self) -> "RssSampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _cache_for(project: Path, key: str) -> Path:
    if key == "minilm":
        return project / "models" / "fastembed"
    return project / "data" / "model-cache-shootout"


def _snapshot_path(cache: Path, spec: SemanticSpec) -> Path:
    repository = cache / ("models--" + spec.model_source.replace("/", "--"))
    snapshot = repository / "snapshots" / spec.model_revision
    if snapshot.is_dir():
        return snapshot
    repository = cache / (
        "models--" + spec.model_source.replace("/", "--").lower()
    )
    return repository / "snapshots" / spec.model_revision


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _thumbnail_paths(catalog_path: Path) -> dict[str, str]:
    connection = sqlite3.connect(catalog_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return {
            image_id: relative_path
            for image_id, relative_path in connection.execute(
                """
                SELECT df.image_id, df.relative_path
                FROM derived_files df
                WHERE df.kind='thumbnail' AND df.id=(
                    SELECT df2.id FROM derived_files df2
                    WHERE df2.image_id=df.image_id AND df2.kind='thumbnail'
                    ORDER BY df2.id DESC LIMIT 1
                )
                """
            )
        }
    finally:
        connection.close()


def _image_content_hashes(catalog_path: Path) -> dict[str, str]:
    connection = sqlite3.connect(catalog_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return dict(connection.execute("SELECT id, content_sha256 FROM images"))
    finally:
        connection.close()


def _result_row(rank: int, result: object) -> dict[str, object]:
    return {
        "rank": rank,
        "image_id": result.image_id,
        "relative_path": result.relative_path,
        "subtitle_text": result.subtitle_text,
        "score": result.score,
    }


def _benchmark_model(
    project: Path,
    key: str,
    spec: SemanticSpec,
    queries: list[str],
    limit: int,
    top_k: int,
    latency_runs: int,
    rebuild: bool,
) -> dict[str, object]:
    cache = _cache_for(project, key)
    index_path = project / "data" / f"semantic-model-shootout-{key}.sqlite3"
    print(f"\n[{MODEL_LABELS[key]}] loading {spec.model_name}", flush=True)
    with RssSampler() as memory:
        load_started = time.perf_counter()
        embedder = FastEmbedder(cache, spec)
        load_seconds = time.perf_counter() - load_started
        rss_after_load = psutil.Process().memory_info().rss
        print(f"[{MODEL_LABELS[key]}] building {limit}-subtitle index", flush=True)
        build = build_trial_index(
            project / "data" / "image-finder.sqlite3",
            index_path,
            embedder,
            spec,
            limit=limit,
            rebuild=rebuild,
        )
        rss_after_build = psutil.Process().memory_info().rss
        semantic_search(index_path, queries[0], embedder, spec, limit=top_k)
        latencies: list[float] = []
        query_results: list[dict[str, object]] = []
        for query in queries:
            first_results = None
            query_latencies = []
            for _ in range(latency_runs):
                started = time.perf_counter()
                results = semantic_search(index_path, query, embedder, spec, limit=top_k)
                elapsed = time.perf_counter() - started
                query_latencies.append(elapsed)
                latencies.append(elapsed)
                if first_results is None:
                    first_results = results
            query_results.append(
                {
                    "query": query,
                    "latencies_seconds": query_latencies,
                    "results": [
                        _result_row(rank, item)
                        for rank, item in enumerate(first_results or [], 1)
                    ],
                }
            )
            top = (first_results or [None])[0]
            if top is not None:
                print(
                    f"[{MODEL_LABELS[key]}] {query!r} -> "
                    f"{top.subtitle_text!r} ({top.score:.4f})",
                    flush=True,
                )

    snapshot = _snapshot_path(cache, spec)
    performance = {
        "model_load_seconds": load_seconds,
        "index_build_total_seconds": build.total_seconds,
        "embedding_seconds": build.embedding_seconds,
        "embedding_subtitles_per_second": (
            build.sample_size / build.embedding_seconds
            if build.embedding_seconds
            else None
        ),
        "warm_query_median_seconds": statistics.median(latencies),
        "warm_query_mean_seconds": statistics.mean(latencies),
        "warm_query_p95_seconds": _percentile(latencies, 0.95),
        "latency_measurements": len(latencies),
        "raw_vector_bytes": build.sample_size * spec.dimensions * 4,
        "sqlite_index_bytes": index_path.stat().st_size,
        "model_snapshot_bytes": _directory_bytes(snapshot),
        "rss_start_bytes": memory.start_bytes,
        "rss_after_load_bytes": rss_after_load,
        "rss_after_build_bytes": rss_after_build,
        "rss_observed_peak_bytes": memory.peak_bytes,
        "rss_observed_peak_delta_bytes": memory.peak_bytes - memory.start_bytes,
        "rss_note": "50 ms process working-set samples; observed peak, not a hard maximum",
    }
    return {
        "key": key,
        "label": MODEL_LABELS[key],
        "spec": spec.as_dict(),
        "build": asdict(build),
        "index_relative_path": index_path.relative_to(project).as_posix(),
        "model_snapshot_relative_path": snapshot.relative_to(project).as_posix(),
        "performance": performance,
        "queries": query_results,
    }


def _format_size(value: int) -> str:
    return f"{value / 1024 / 1024:.1f} MB"


def _format_ms(value: float) -> str:
    return f"{value * 1000:.1f} ms"


def _render_html(
    report: dict[str, object],
    thumbnails: dict[str, str],
    image_content_hashes: dict[str, str],
) -> str:
    models = report["models"]
    review_items = []
    performance_rows = []
    for model in models:
        perf = model["performance"]
        performance_rows.append(
            "<tr>"
            f"<th>{html.escape(model['label'])}</th>"
            f"<td>{model['spec']['dimensions']}</td>"
            f"<td>{_format_size(perf['model_snapshot_bytes'])}</td>"
            f"<td>{perf['model_load_seconds']:.3f} s</td>"
            f"<td>{perf['embedding_seconds']:.3f} s</td>"
            f"<td>{perf['embedding_subtitles_per_second']:.2f}/s</td>"
            f"<td>{_format_ms(perf['warm_query_median_seconds'])}</td>"
            f"<td>{_format_ms(perf['warm_query_p95_seconds'])}</td>"
            f"<td>{_format_size(perf['sqlite_index_bytes'])}</td>"
            "</tr>"
        )

    query_sections = []
    queries = report["query_set"]["queries"]
    for query_index, query_metadata in enumerate(queries):
        columns = []
        for model in models:
            evaluation = model["queries"][query_index]
            cards = []
            for item in evaluation["results"]:
                thumb = thumbnails.get(item["image_id"])
                image = (
                    f'<img loading="lazy" src="{quote(thumb)}" alt="">'
                    if thumb
                    else '<div class="missing">No thumbnail</div>'
                )
                review_key = (
                    f"q{query_index}|{model['key']}|"
                    f"{model['spec']['model_revision']}|{item['image_id']}"
                )
                review_items.append(
                    {
                        "review_key": review_key,
                        "query": query_metadata["query"],
                        "model_key": model["key"],
                        "model_identifier": model["spec"]["model_name"],
                        "model_revision": model["spec"]["model_revision"],
                        "image_id": item["image_id"],
                        "image_content_sha256": image_content_hashes[item["image_id"]],
                        "relative_path": item["relative_path"],
                        "ocr_subtitle": item["subtitle_text"],
                        "rank": item["rank"],
                        "similarity_score": item["score"],
                    }
                )
                choice_name = html.escape(review_key, quote=True)
                choices = "".join(
                    f'<label><input type="radio" name="{choice_name}" '
                    f'data-review-key="{choice_name}" value="{value}"> {label}</label>'
                    for value, label in (
                        ("useful", "Useful"),
                        ("maybe", "Maybe"),
                        ("not_useful", "Not useful"),
                    )
                )
                cards.append(
                    '<article class="result">'
                    f"{image}"
                    '<div class="result-body">'
                    f'<div class="rank">#{item["rank"]} · cosine {item["score"]:.4f}</div>'
                    f'<div class="subtitle">{html.escape(item["subtitle_text"])}</div>'
                    f'<div class="path">{html.escape(item["relative_path"])}</div>'
                    f'<div class="choices">{choices}</div>'
                    "</div></article>"
                )
            columns.append(
                '<section class="model-column">'
                f"<h3>{html.escape(model['label'])}</h3>"
                + "".join(cards)
                + "</section>"
            )
        query_sections.append(
            '<section class="query">'
            f'<h2>{html.escape(query_metadata["query"])}</h2>'
            f'<p class="category">{html.escape(query_metadata["category"])}</p>'
            f'<div class="model-grid">{"".join(columns)}</div>'
            "</section>"
        )

    generated = html.escape(report["generated_at"])
    manifest = html.escape(report["sample"]["manifest_sha256"])
    review_context = {
        "evaluation_report_version": report["format_version"],
        "report_generated_at": report["generated_at"],
        "query_set_version": report["query_set"]["version"],
        "sample_manifest_sha256": report["sample"]["manifest_sha256"],
        "total_result_count": len(review_items),
        "items": review_items,
    }
    context_json = json.dumps(
        review_context, ensure_ascii=False, separators=(",", ":")
    ).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Finder semantic model shootout</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, Segoe UI, sans-serif; background:#111318; color:#edf0f7; }}
body {{ margin:0; }} main {{ max-width:1800px; margin:auto; padding:28px; }}
h1 {{ margin-bottom:6px; }} .intro,.category,.meta {{ color:#aeb7ca; }}
.summary {{ width:100%; border-collapse:collapse; margin:24px 0 36px; font-size:14px; }}
.summary th,.summary td {{ padding:10px; border:1px solid #343b49; text-align:right; }}
.summary th:first-child {{ text-align:left; }} .summary thead th {{ background:#202532; }}
.query {{ border-top:2px solid #343b49; padding:24px 0 34px; }} .query h2 {{ margin:0; }}
.category {{ margin:4px 0 16px; }} .model-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }}
.model-column {{ background:#191d26; border:1px solid #303746; border-radius:12px; padding:12px; }}
.model-column h3 {{ margin:4px 4px 12px; color:#9fc4ff; }}
.result {{ display:grid; grid-template-columns:160px 1fr; min-height:100px; overflow:hidden; background:#232936; border-radius:9px; margin:10px 0; }}
.result img,.missing {{ width:160px; height:100px; object-fit:cover; background:#0a0b0e; }}
.missing {{ display:grid; place-items:center; color:#717a8d; }} .result-body {{ padding:9px 11px; min-width:0; }}
.rank {{ color:#aeb7ca; font-size:12px; }} .subtitle {{ font-size:17px; margin:3px 0 5px; }}
.path {{ color:#8993a8; font-size:11px; overflow-wrap:anywhere; }} .choices {{ display:flex; gap:9px; margin-top:8px; font-size:11px; flex-wrap:wrap; }}
.note {{ padding:12px 14px; background:#282313; border:1px solid #685c29; border-radius:8px; }}
.review-toolbar {{ position:sticky; top:0; z-index:10; margin:18px 0 24px; padding:14px; border:1px solid #40506b; border-radius:10px; background:rgba(25,29,38,.97); box-shadow:0 10px 28px rgba(0,0,0,.3); }}
.review-counts {{ display:flex; flex-wrap:wrap; gap:8px 18px; margin-bottom:10px; }}
.review-counts strong {{ color:#fff; }} .review-actions {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
.review-actions button {{ border:1px solid #5d6d8d; border-radius:7px; padding:8px 11px; background:#2b3549; color:#fff; cursor:pointer; }}
.review-actions button:hover {{ background:#35435d; }} .review-actions .danger {{ border-color:#8b4e55; background:#512d33; }}
#review-progress {{ width:min(520px,100%); height:12px; }} #review-status {{ margin:8px 0 0; color:#9fd6ad; font-size:13px; }}
#review-status.error {{ color:#ffb0ad; }} .visually-hidden {{ position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }}
@media(max-width:1100px) {{ .model-grid {{ grid-template-columns:1fr; }} }}
</style></head><body><main>
<h1>Semantic subtitle model shootout</h1>
<p class="intro">Same deterministic 1,000 OCR subtitles and same fixed 16 queries. Scores are cosine similarities within each model only—not probabilities and not comparable across model families.</p>
<p class="note">Your ratings save automatically in this browser. Export JSON periodically for backup or to continue on another browser or machine. No automated relevance labels have been invented.</p>
<p class="meta">Generated {generated} · sample manifest <code>{manifest}</code></p>
<section class="review-toolbar" aria-label="Review progress and controls">
  <div class="review-counts">
    <span>Reviewed <strong id="reviewed-count">0</strong> / {len(review_items)}</span>
    <span>Remaining <strong id="remaining-count">{len(review_items)}</strong></span>
    <span>Useful <strong id="useful-count">0</strong></span>
    <span>Maybe <strong id="maybe-count">0</strong></span>
    <span>Not useful <strong id="not-useful-count">0</strong></span>
  </div>
  <div class="review-actions">
    <progress id="review-progress" value="0" max="{len(review_items)}"></progress>
    <button id="export-review" type="button">Export review results (0)</button>
    <button id="import-review" type="button">Import review results</button>
    <input class="visually-hidden" id="import-review-file" type="file" accept="application/json,.json">
    <button class="danger" id="reset-review" type="button">Clear review</button>
  </div>
  <p id="review-status" role="status" aria-live="polite"></p>
</section>
<table class="summary"><thead><tr><th>Model</th><th>Dims</th><th>Package</th><th>Load</th><th>Embed 1,000</th><th>Throughput</th><th>Query median</th><th>Query p95</th><th>SQLite</th></tr></thead>
<tbody>{''.join(performance_rows)}</tbody></table>
{''.join(query_sections)}
<script id="review-context" type="application/json">{context_json}</script>
<script src="semantic_review.js"></script>
</main></body></html>"""


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--limit", type=int, default=MAX_EVALUATION_IMAGES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--latency-runs", type=int, default=3)
    parser.add_argument(
        "--render-existing",
        action="store_true",
        help="regenerate only the static HTML from the existing result JSON",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    catalog_path = project / "data" / "image-finder.sqlite3"
    if args.render_existing:
        output = project / "results" / "semantic_model_shootout.json"
        report = json.loads(output.read_text(encoding="utf-8"))
        html_output = project / "semantic-model-shootout.html"
        html_output.write_text(
            _render_html(
                report,
                _thumbnail_paths(catalog_path),
                _image_content_hashes(catalog_path),
            ),
            encoding="utf-8",
        )
        print(f"HTML={html_output}")
        print("Existing result JSON reused; no models loaded and no rankings changed.")
        return
    if args.limit != MAX_EVALUATION_IMAGES:
        raise SystemExit("the controlled shootout requires exactly 1,000 subtitles")
    if args.top_k < 5:
        raise SystemExit("the controlled shootout must preserve at least five results")
    if args.latency_runs < 1:
        raise SystemExit("latency-runs must be positive")

    query_document = json.loads(
        (project / "semantic-evaluation-queries.json").read_text(encoding="utf-8")
    )
    queries = [item["query"] for item in query_document["queries"]]
    if len(queries) != 16:
        raise RuntimeError("the fixed evaluation query set no longer contains 16 queries")

    sample = representative_subtitle_sample(catalog_path, args.limit)
    baseline_report = json.loads(
        (project / "results" / "semantic_evaluation_1000.json").read_text(
            encoding="utf-8"
        )
    )
    expected_manifest = baseline_report["build"]["manifest_sha256"]
    manifest = sample_manifest_hash(sample)
    if manifest != expected_manifest:
        raise RuntimeError(
            "current deterministic sample differs from the completed baseline: "
            f"expected {expected_manifest}, found {manifest}"
        )

    models = []
    for key, spec in model_shootout_specs().items():
        models.append(
            _benchmark_model(
                project,
                key,
                spec,
                queries,
                args.limit,
                args.top_k,
                args.latency_runs,
                args.rebuild,
            )
        )
        gc.collect()

    report = {
        "format_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": "Controlled local CPU subtitle-embedding model comparison",
        "sample": {
            "size": len(sample),
            "manifest_sha256": manifest,
            "selection": "same top-level proportional deterministic sample as semantic_evaluation_1000.json",
            "distribution": sample_distribution(sample),
        },
        "query_set": query_document,
        "conditions": {
            "source": "OCR subtitle text only",
            "inference": "local CPU ONNX Runtime through FastEmbed",
            "latency_runs_per_query": args.latency_runs,
            "warmup_queries_per_model": 1,
            "score_warning": "Cosine similarities are not probabilities and are not calibrated across model families.",
        },
        "models": models,
    }
    output = project / "results" / "semantic_model_shootout.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    html_output = project / "semantic-model-shootout.html"
    html_output.write_text(
        _render_html(
            report,
            _thumbnail_paths(catalog_path),
            _image_content_hashes(catalog_path),
        ),
        encoding="utf-8",
    )
    print(f"\nJSON={output}")
    print(f"HTML={html_output}")
    for model in models:
        print(model["label"], json.dumps(model["performance"], indent=2))


if __name__ == "__main__":
    main()
