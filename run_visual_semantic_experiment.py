"""Run one bounded SigLIP 2 experiment over human-reviewed image identities."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sqlite3
import statistics
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import psutil

from image_finder.visual_semantic import (
    Siglip2OnnxEmbedder,
    VisualSpec,
    build_visual_index,
    rank_visual_index,
    review_pair_labels,
    reviewed_image_candidates,
)


FOCUS_QUERIES = {
    "傻眼",
    "很尷尬，不知道該說什麼",
    "很無奈",
    "被嚇到",
    "awkward reaction",
    "安慰心情不好的人",
    "吐槽朋友",
    "朋友講了很蠢的話",
    "鼓勵別人振作",
    "拒絕別人的要求",
    "生氣警告別人",
}


class RssSampler:
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _thumbnails(catalog_path: Path) -> dict[str, str]:
    connection = sqlite3.connect(catalog_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return dict(
            connection.execute(
                """
                SELECT df.image_id, df.relative_path FROM derived_files df
                WHERE df.kind='thumbnail' AND df.id=(
                    SELECT df2.id FROM derived_files df2
                    WHERE df2.image_id=df.image_id AND df2.kind='thumbnail'
                    ORDER BY df2.id DESC LIMIT 1
                )
                """
            )
        )
    finally:
        connection.close()


def _present_image_count(catalog_path: Path) -> int:
    connection = sqlite3.connect(catalog_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return int(
            connection.execute(
                "SELECT COUNT(DISTINCT image_id) FROM file_locations WHERE is_present=1"
            ).fetchone()[0]
        )
    finally:
        connection.close()


def _validate_review(review: dict, shootout: dict) -> dict[str, object]:
    expected = {}
    for query_index, query in enumerate(shootout["query_set"]["queries"]):
        for model in shootout["models"]:
            for result in model["queries"][query_index]["results"]:
                key = (
                    f"q{query_index}|{model['key']}|"
                    f"{model['spec']['model_revision']}|{result['image_id']}"
                )
                expected[key] = (
                    query["query"],
                    model["key"],
                    model["spec"]["model_name"],
                    model["spec"]["model_revision"],
                    result["image_id"],
                    result["relative_path"],
                    result["subtitle_text"],
                    result["rank"],
                    result["score"],
                )
    actual = {
        item["review_key"]: (
            item["query"],
            item["model_key"],
            item["model_identifier"],
            item["model_revision"],
            item["image_id"],
            item["relative_path"],
            item["ocr_subtitle"],
            item["rank"],
            item["similarity_score"],
        )
        for item in review["ratings"]
    }
    checks = {
        "reviewed_result_count": review["reviewed_result_count"],
        "total_result_count": review["total_result_count"],
        "query_count": len(shootout["query_set"]["queries"]),
        "model_count": len(shootout["models"]),
        "expected_rows": len(expected),
        "unique_review_keys": len(actual),
        "keys_match": set(expected) == set(actual),
        "payload_matches": sum(actual.get(key) == value for key, value in expected.items()),
        "evaluation_version_matches": (
            review["evaluation_report_version"] == shootout["format_version"]
        ),
        "query_set_matches": (
            review["query_set_version"] == shootout["query_set"]["version"]
        ),
        "sample_manifest_matches": (
            review["sample_manifest_sha256"]
            == shootout["sample"]["manifest_sha256"]
        ),
    }
    if not (
        checks["reviewed_result_count"] == checks["total_result_count"] == 240
        and checks["keys_match"]
        and checks["payload_matches"] == 240
        and checks["evaluation_version_matches"]
        and checks["query_set_matches"]
        and checks["sample_manifest_matches"]
    ):
        raise ValueError(f"completed review validation failed: {checks}")
    return checks


def _source_state(candidates: list) -> dict[str, tuple[int, int]]:
    return {
        item.image_id: (item.source_path.stat().st_size, item.source_path.stat().st_mtime_ns)
        for item in candidates
    }


def _existing_rating_rows(review: dict) -> dict[tuple[str, str], list[str]]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item in review["ratings"]:
        grouped[(item["query"], item["image_id"])].append(item["human_rating"])
    return grouped


def _rank_summary(
    queries: list[dict], labels: dict[tuple[str, str], str]
) -> dict[str, object]:
    by_label: dict[str, list[int]] = defaultdict(list)
    top_counts = {5: Counter(), 10: Counter(), 15: Counter()}
    recoveries_at_5 = 0
    recoveries_at_10 = 0
    shared_useful_at_5 = 0
    minilm_useful_missed_at_5 = 0
    for query in queries:
        name = query["query"]
        visual_rank = {item["image_id"]: item["rank"] for item in query["results_all"]}
        for (label_query, image_id), label in labels.items():
            if label_query == name:
                by_label[label].append(visual_rank[image_id])
        for cutoff in top_counts:
            for item in query["results_all"][:cutoff]:
                top_counts[cutoff][item["existing_human_rating"]] += 1

        minilm_useful = {
            item["image_id"]
            for item in query["minilm_top5"]
            if item["existing_human_rating"] == "useful"
        }
        reviewed_useful = {
            image_id
            for (label_query, image_id), label in labels.items()
            if label_query == name and label == "useful"
        }
        visual_5 = {item["image_id"] for item in query["results_all"][:5]}
        visual_10 = {item["image_id"] for item in query["results_all"][:10]}
        recoveries_at_5 += len((reviewed_useful - minilm_useful) & visual_5)
        recoveries_at_10 += len((reviewed_useful - minilm_useful) & visual_10)
        shared_useful_at_5 += len(minilm_useful & visual_5)
        minilm_useful_missed_at_5 += len(minilm_useful - visual_5)

    rank_metrics = {}
    for label, ranks in sorted(by_label.items()):
        rank_metrics[label] = {
            "pair_count": len(ranks),
            "mean_rank": statistics.mean(ranks),
            "median_rank": statistics.median(ranks),
            "within_top_5": sum(rank <= 5 for rank in ranks),
            "within_top_10": sum(rank <= 10 for rank in ranks),
            "within_top_15": sum(rank <= 15 for rank in ranks),
        }
    return {
        "reviewed_pair_rank_metrics": rank_metrics,
        "visual_top_k_label_counts": {
            str(cutoff): dict(counts) for cutoff, counts in top_counts.items()
        },
        "reviewed_useful_not_in_minilm_recovered_by_visual_top_5": recoveries_at_5,
        "reviewed_useful_not_in_minilm_recovered_by_visual_top_10": recoveries_at_10,
        "shared_minilm_visual_useful_in_both_top_5": shared_useful_at_5,
        "minilm_useful_outside_visual_top_5": minilm_useful_missed_at_5,
    }


def _render_html(report: dict, thumbnails: dict[str, str]) -> str:
    selected_keys = set(report["supplemental_review"]["review_keys"])
    review_items = []
    sections = []
    for query_index, query in enumerate(report["queries"]):
        cards = []
        for result in query["results_top_10"]:
            label = result["existing_human_rating"]
            label_text = {
                "useful": "Reviewed Useful",
                "maybe": "Reviewed Maybe",
                "not_useful": "Reviewed Not useful",
                "conflict": "Reviewed conflict",
                "unreviewed": "Unreviewed",
            }[label]
            review_key = (
                f"visual|q{query_index}|{report['model']['spec']['onnx_revision']}|"
                f"{result['image_id']}"
            )
            choices = ""
            if review_key in selected_keys:
                review_item = {
                    "review_key": review_key,
                    "query": query["query"],
                    "model_key": "siglip2-visual-int8",
                    "model_identifier": report["model"]["spec"]["model_name"],
                    "model_revision": report["model"]["spec"]["onnx_revision"],
                    "image_id": result["image_id"],
                    "image_content_sha256": result["content_sha256"],
                    "relative_path": result["relative_path"],
                    "ocr_subtitle": result["subtitle_text"],
                    "rank": result["rank"],
                    "similarity_score": result["score"],
                }
                review_items.append(review_item)
                escaped_key = html.escape(review_key, quote=True)
                choices = '<div class="choices">' + "".join(
                    f'<label><input type="radio" name="{escaped_key}" '
                    f'data-review-key="{escaped_key}" value="{value}"> {text}</label>'
                    for value, text in (
                        ("useful", "Useful"),
                        ("maybe", "Maybe"),
                        ("not_useful", "Not useful"),
                    )
                ) + "</div>"
            thumbnail = thumbnails[result["image_id"]]
            cards.append(
                '<article class="result">'
                f'<img loading="lazy" src="{quote(thumbnail)}" alt="">'
                '<div class="body">'
                f'<div class="rank">#{result["rank"]} · cosine {result["score"]:.4f}</div>'
                f'<div class="badge {label}">{label_text}</div>'
                f'<div class="subtitle">{html.escape(result["subtitle_text"])}</div>'
                f'<div class="path">{html.escape(result["relative_path"])}</div>'
                f"{choices}</div></article>"
            )
        sections.append(
            '<section class="query">'
            f'<h2>{html.escape(query["query"])}</h2>'
            f'<p class="category">{html.escape(query["category"])}</p>'
            f'<div class="grid">{"".join(cards)}</div></section>'
        )

    context = {
        "evaluation_report_version": "visual-semantic-reviewed-set-v1",
        "report_generated_at": report["generated_at"],
        "query_set_version": report["query_set"]["version"],
        "sample_manifest_sha256": report["selection"]["manifest_sha256"],
        "report_slug": "visual-semantic-supplemental",
        "total_result_count": len(review_items),
        "items": review_items,
    }
    context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    total = len(review_items)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Image Finder visual-semantic review</title>
<style>
:root{{color-scheme:dark;font-family:Inter,Segoe UI,sans-serif;background:#111318;color:#edf0f7}}
body{{margin:0}}main{{max-width:1680px;margin:auto;padding:28px}}.intro,.category,.path{{color:#aeb7ca}}
.toolbar{{position:sticky;top:0;z-index:5;background:#191d26f5;border:1px solid #40506b;border-radius:10px;padding:14px;margin:20px 0}}
.counts,.actions{{display:flex;gap:10px 18px;flex-wrap:wrap;align-items:center}}button{{background:#2b3549;color:white;border:1px solid #5d6d8d;border-radius:7px;padding:8px 11px}}.danger{{background:#512d33}}
.query{{border-top:2px solid #343b49;padding:24px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:13px}}
.result{{background:#202530;border-radius:10px;overflow:hidden}}.result img{{width:100%;aspect-ratio:16/10;object-fit:contain;background:#08090c}}.body{{padding:10px}}
.rank{{font-size:12px;color:#aeb7ca}}.subtitle{{font-size:16px;margin:7px 0}}.path{{font-size:11px;overflow-wrap:anywhere}}.badge{{display:inline-block;margin-top:5px;padding:3px 7px;border-radius:999px;font-size:11px;background:#4a4f5a}}.useful{{background:#235b40}}.maybe{{background:#66551d}}.not_useful{{background:#6a3035}}.conflict{{background:#684277}}.choices{{display:flex;gap:8px;flex-wrap:wrap;margin-top:9px;font-size:12px}}#review-status{{color:#9fd6ad}}#review-status.error{{color:#ffb0ad}}.visually-hidden{{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}}
</style></head><body><main>
<h1>Bounded visual-semantic experiment</h1>
<p class="intro">SigLIP 2 visual-only rankings over the 148 images already represented in the completed text-model review. Existing labels apply only to exact reviewed query–image pairs. Every other result remains Unreviewed.</p>
<p class="intro">The rating controls appear only on {total} selected, previously unreviewed results from the priority queries. Ratings save locally and export separately; they do not alter the completed text review.</p>
<section class="toolbar"><div class="counts"><span>Reviewed <strong id="reviewed-count">0</strong> / {total}</span><span>Remaining <strong id="remaining-count">{total}</strong></span><span>Useful <strong id="useful-count">0</strong></span><span>Maybe <strong id="maybe-count">0</strong></span><span>Not useful <strong id="not-useful-count">0</strong></span></div>
<div class="actions"><progress id="review-progress" value="0" max="{total}"></progress><button id="export-review">Export review results (0)</button><button id="import-review">Import review results</button><input class="visually-hidden" id="import-review-file" type="file" accept="application/json,.json"><button class="danger" id="reset-review">Clear review</button></div><p id="review-status"></p></section>
{"".join(sections)}
<script id="review-context" type="application/json">{context_json}</script><script src="semantic_review.js"></script>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--latency-runs", type=int, default=3)
    args = parser.parse_args()
    if args.top_k < 15:
        parser.error("--top-k must be at least 15")
    if args.image_batch_size < 1:
        parser.error("--image-batch-size must be positive")

    project = Path(__file__).resolve().parent
    results_dir = project / "results"
    catalog_path = project / "data" / "image-finder.sqlite3"
    review_path = next(
        results_dir.glob(
            "image-finder-semantic-model-shootout-*-complete-240-of-240-*.json"
        )
    )
    shootout_path = results_dir / "semantic_model_shootout.json"
    index_path = project / "data" / "visual-semantic-reviewed-set.sqlite3"
    model_dir = project / "models" / "siglip2-base-patch16-224-onnx-int8"
    output_path = results_dir / "visual_semantic_reviewed_set.json"
    html_path = project / "visual-semantic-reviewed-set.html"

    previous_report = None
    if output_path.is_file():
        try:
            previous_report = json.loads(output_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous_report = None

    review = json.loads(review_path.read_text(encoding="utf-8"))
    shootout = json.loads(shootout_path.read_text(encoding="utf-8"))
    validation = _validate_review(review, shootout)
    candidates = reviewed_image_candidates(review, catalog_path)
    if len(candidates) != 148:
        raise RuntimeError(f"expected exactly 148 reviewed images, found {len(candidates)}")
    state_before = _source_state(candidates)
    labels = review_pair_labels(review)
    rating_rows = _existing_rating_rows(review)
    label_counts = Counter(labels.values())
    queries = shootout["query_set"]["queries"]
    minilm = next(model for model in shootout["models"] if model["key"] == "minilm")
    minilm_ratings = {
        (item["query"], item["image_id"]): item["human_rating"]
        for item in review["ratings"]
        if item["model_key"] == "minilm"
    }

    spec = VisualSpec()
    with RssSampler() as memory:
        load_started = time.perf_counter()
        embedder = Siglip2OnnxEmbedder(
            model_dir, spec, image_batch_size=args.image_batch_size
        )
        model_load_seconds = time.perf_counter() - load_started
        rss_after_load = psutil.Process().memory_info().rss
        build = build_visual_index(
            index_path, candidates, embedder, spec, rebuild=args.rebuild
        )
        rss_after_build = psutil.Process().memory_info().rss
        embedder.release_image_model()
        rss_after_image_session_release = psutil.Process().memory_info().rss

        warmup_started = time.perf_counter()
        embedder.embed_texts(["warm-up"])
        text_warmup_seconds = time.perf_counter() - warmup_started

        query_rows = []
        encoding_latencies = []
        ranking_latencies = []
        end_to_end_latencies = []
        for query_index, metadata in enumerate(queries):
            query = metadata["query"]
            first_results = None
            per_query = {"encoding": [], "ranking": [], "end_to_end": []}
            for _ in range(args.latency_runs):
                overall_started = time.perf_counter()
                encode_started = time.perf_counter()
                vector = embedder.embed_texts([query])[0]
                encode_seconds = time.perf_counter() - encode_started
                rank_started = time.perf_counter()
                ranked = rank_visual_index(
                    index_path, vector, spec, limit=len(candidates)
                )
                rank_seconds = time.perf_counter() - rank_started
                total_seconds = time.perf_counter() - overall_started
                if first_results is None:
                    first_results = ranked
                per_query["encoding"].append(encode_seconds)
                per_query["ranking"].append(rank_seconds)
                per_query["end_to_end"].append(total_seconds)
                encoding_latencies.append(encode_seconds)
                ranking_latencies.append(rank_seconds)
                end_to_end_latencies.append(total_seconds)

            result_rows = []
            for rank, item in enumerate(first_results or [], 1):
                pair = (query, item.image_id)
                existing = labels.get(pair, "unreviewed")
                result_rows.append(
                    {
                        "rank": rank,
                        "image_id": item.image_id,
                        "content_sha256": item.content_sha256,
                        "relative_path": item.relative_path,
                        "subtitle_text": item.subtitle_text,
                        "score": item.score,
                        "existing_human_rating": existing,
                        "existing_human_ratings": rating_rows.get(pair, []),
                    }
                )
            baseline = []
            for item in minilm["queries"][query_index]["results"]:
                baseline.append(
                    {
                        **item,
                        "existing_human_rating": minilm_ratings[(query, item["image_id"])],
                    }
                )
            query_rows.append(
                {
                    **metadata,
                    "latencies_seconds": per_query,
                    "results_top_10": result_rows[:10],
                    "results_top_15": result_rows[: args.top_k],
                    "results_all": result_rows,
                    "minilm_top5": baseline,
                }
            )

    if _source_state(candidates) != state_before:
        raise RuntimeError("a source image changed while the visual experiment ran")
    for item in candidates:
        if _file_sha256(item.source_path) != item.content_sha256:
            raise RuntimeError(f"source content hash changed: {item.relative_path}")

    supplemental_keys = []
    for query_index, query in enumerate(query_rows):
        if query["query"] not in FOCUS_QUERIES:
            continue
        chosen = 0
        for item in query["results_top_10"]:
            if item["existing_human_rating"] != "unreviewed":
                continue
            supplemental_keys.append(
                f"visual|q{query_index}|{spec.onnx_revision}|{item['image_id']}"
            )
            chosen += 1
            if chosen == 3:
                break

    full_count = _present_image_count(catalog_path)
    performance = {
        "model_load_seconds": model_load_seconds,
        "index_reused": build.reused,
        "image_embedding_seconds": build.embedding_seconds,
        "total_index_build_seconds": build.total_seconds,
        "images_per_second": (
            build.image_count / build.embedding_seconds
            if build.embedding_seconds
            else None
        ),
        "text_warmup_seconds": text_warmup_seconds,
        "text_encoding_median_seconds": statistics.median(encoding_latencies),
        "text_encoding_p95_seconds": _percentile(encoding_latencies, 0.95),
        "ranking_median_seconds": statistics.median(ranking_latencies),
        "ranking_p95_seconds": _percentile(ranking_latencies, 0.95),
        "end_to_end_query_median_seconds": statistics.median(end_to_end_latencies),
        "end_to_end_query_p95_seconds": _percentile(end_to_end_latencies, 0.95),
        "latency_measurements": len(end_to_end_latencies),
        "embedding_dimensions": spec.dimensions,
        "raw_vector_bytes": len(candidates) * spec.dimensions * 4,
        "sqlite_index_bytes": index_path.stat().st_size,
        "selected_model_files_bytes": _directory_bytes(model_dir),
        "rss_start_bytes": memory.start_bytes,
        "rss_after_load_bytes": rss_after_load,
        "rss_after_build_bytes": rss_after_build,
        "rss_after_image_session_release_bytes": rss_after_image_session_release,
        "rss_observed_peak_bytes": memory.peak_bytes,
        "rss_observed_peak_delta_bytes": memory.peak_bytes - memory.start_bytes,
        "rss_note": "50 ms process working-set samples; observed peak, not a hard maximum",
        "full_present_image_count": full_count,
        "estimated_full_library_embedding_seconds": (
            full_count * build.embedding_seconds / build.image_count
            if build.embedding_seconds
            else None
        ),
        "estimated_full_library_raw_vector_bytes": full_count * spec.dimensions * 4,
    }
    if (
        build.reused
        and previous_report
        and previous_report.get("selection", {}).get("manifest_sha256")
        == build.manifest_sha256
        and previous_report.get("model", {}).get("spec") == spec.as_dict()
        and not previous_report.get("performance", {}).get("index_reused", True)
    ):
        performance["last_fresh_build_performance"] = previous_report["performance"]
    report = {
        "format_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "purpose": "bounded visual-only retrieval over completed-review image identities",
        "validation": validation,
        "source_review": {
            "relative_path": review_path.relative_to(project).as_posix(),
            "sha256": _file_sha256(review_path),
            "unique_reviewed_query_image_pairs": len(labels),
            "consistent_label_counts": dict(label_counts),
        },
        "query_set": shootout["query_set"],
        "selection": {
            "policy": "deduplicate completed-review results by stable content hash",
            "unique_images": len(candidates),
            "unique_content_hashes": len({item.content_sha256 for item in candidates}),
            "manifest_sha256": build.manifest_sha256,
            "source_images_modified": False,
        },
        "model": {
            "key": "siglip2-visual-int8",
            "label": "SigLIP 2 Base Patch16 224 int8 ONNX",
            "spec": spec.as_dict(),
        },
        "build": asdict(build),
        "performance": performance,
        "evaluation": _rank_summary(query_rows, labels),
        "supplemental_review": {
            "policy": "first three unreviewed visual Top-10 results for each priority query",
            "priority_queries": sorted(FOCUS_QUERIES),
            "result_count": len(supplemental_keys),
            "review_keys": supplemental_keys,
        },
        "queries": query_rows,
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html_path.write_text(_render_html(report, _thumbnails(catalog_path)), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "html": str(html_path),
                "selection": report["selection"],
                "performance": performance,
                "evaluation": report["evaluation"],
                "supplemental_review_count": len(supplemental_keys),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
