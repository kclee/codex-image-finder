"""Small, local semantic-search experiment over versioned OCR subtitles only."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import sqlite3
import struct
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from .text_search import query_variants


MAX_TRIAL_IMAGES = 50
MAX_EVALUATION_IMAGES = 1000
INDEX_SCHEMA_VERSION = 1
PIPELINE_VERSION = "ocr-subtitle-semantic-v1"
EVALUATION_PIPELINE_VERSION = "ocr-subtitle-semantic-stratified-v2"
SHOOTOUT_PIPELINE_VERSION = "ocr-subtitle-semantic-model-shootout-v1"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_SOURCE = "qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
MODEL_REVISION = "faf4aa4225822f3bc6376869cb1164e8e3feedd0"
EMBEDDING_DIMENSIONS = 384
E5_SMALL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MPNET_REVISION = "e5d116277351513fd260955ece953ecddde7046e"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SemanticSpec:
    engine_name: str = "FastEmbed"
    engine_version: str = "0.8.0"
    runtime_name: str = "onnxruntime"
    runtime_version: str = "1.29.0"
    model_name: str = MODEL_NAME
    model_source: str = MODEL_SOURCE
    model_revision: str = MODEL_REVISION
    pipeline_version: str = PIPELINE_VERSION
    dimensions: int = EMBEDDING_DIMENSIONS
    index_schema_version: int = INDEX_SCHEMA_VERSION
    parameters: dict[str, object] = field(
        default_factory=lambda: {
            "source_field": "ocr.subtitle_text",
            "sample_order": "images.content_sha256 ASC",
            "maximum_sample_size": MAX_TRIAL_IMAGES,
            "distance": "cosine",
            "vector_dtype": "float32",
            "l2_normalize": True,
            "pooling": "mean",
            "maximum_tokens": 512,
            "query_prefix": None,
            "document_prefix": None,
        }
    )

    @property
    def version_id(self) -> str:
        return _canonical_hash(self.as_dict())[:24]

    def as_dict(self) -> dict[str, object]:
        return {
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "runtime_name": self.runtime_name,
            "runtime_version": self.runtime_version,
            "model_name": self.model_name,
            "model_source": self.model_source,
            "model_revision": self.model_revision,
            "pipeline_version": self.pipeline_version,
            "dimensions": self.dimensions,
            "index_schema_version": self.index_schema_version,
            "parameters": self.parameters,
        }


@dataclass(frozen=True, slots=True)
class SubtitleCandidate:
    image_id: str
    relative_path: str
    subtitle_text: str
    source_text_sha256: str


@dataclass(frozen=True, slots=True)
class BuildSummary:
    run_id: str
    sample_size: int
    manifest_sha256: str
    selection_seconds: float
    embedding_seconds: float
    total_seconds: float
    reused: bool


@dataclass(frozen=True, slots=True)
class SemanticResult:
    image_id: str
    relative_path: str
    subtitle_text: str
    source_text_sha256: str
    score: float


@dataclass(frozen=True, slots=True)
class HybridResult:
    image_id: str
    relative_path: str
    subtitle_text: str
    source_text_sha256: str
    hybrid_score: float
    semantic_score: float
    lexical_score: float | None


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class FastEmbedder:
    """Lazy FastEmbed adapter; model files always live below the project."""

    def __init__(self, cache_dir: Path, spec: SemanticSpec) -> None:
        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType

        installed = {
            "fastembed": importlib.metadata.version("fastembed"),
            "onnxruntime": importlib.metadata.version("onnxruntime"),
        }
        expected = {
            "fastembed": spec.engine_version,
            "onnxruntime": spec.runtime_version,
        }
        if installed != expected:
            raise RuntimeError(
                f"embedding dependency mismatch: expected {expected}, found {installed}"
            )
        repository = cache_dir / ("models--" + spec.model_source.replace("/", "--"))
        custom_model_id = spec.parameters.get("fastembed_model_id")
        model_file = str(spec.parameters.get("model_file", "onnx/model.onnx"))
        specific_model_path: str | None = None
        fastembed_model_name = spec.model_name
        if custom_model_id:
            snapshot = repository / "snapshots" / spec.model_revision
            if not (snapshot / model_file).is_file():
                raise RuntimeError(
                    "pinned local model snapshot is unavailable: "
                    f"{snapshot / model_file}"
                )
            fastembed_model_name = str(custom_model_id)
            registered = {
                item["model"].casefold() for item in TextEmbedding.list_supported_models()
            }
            if fastembed_model_name.casefold() not in registered:
                pooling_name = str(spec.parameters.get("pooling", "mean")).upper()
                try:
                    pooling = PoolingType[pooling_name]
                except KeyError as error:
                    raise ValueError(f"unsupported pooling mode: {pooling_name}") from error
                TextEmbedding.add_custom_model(
                    model=fastembed_model_name,
                    pooling=pooling,
                    normalization=bool(spec.parameters.get("l2_normalize", True)),
                    sources=ModelSource(hf=spec.model_source),
                    dim=spec.dimensions,
                    model_file=model_file,
                    description="Pinned Image Finder semantic model evaluation",
                    license=str(spec.parameters.get("license", "")),
                )
            specific_model_path = str(snapshot)
        else:
            revision_file = repository / "refs" / "main"
            if not revision_file.is_file():
                # FastEmbed historically lower-cased this cache directory.
                repository = cache_dir / (
                    "models--" + spec.model_source.replace("/", "--").lower()
                )
                revision_file = repository / "refs" / "main"
            if not revision_file.is_file():
                raise RuntimeError(
                    f"local model revision is unavailable: {revision_file}"
                )
            cached_revision = revision_file.read_text(encoding="utf-8").strip()
            if cached_revision != spec.model_revision:
                raise RuntimeError(
                    "model revision mismatch: "
                    f"expected {spec.model_revision}, found {cached_revision}"
                )
        self._model = TextEmbedding(
            model_name=fastembed_model_name,
            cache_dir=str(cache_dir),
            local_files_only=True,
            specific_model_path=specific_model_path,
        )

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return list(self._model.embed(list(texts)))


def _latest_subtitle_rows(catalog_path: Path) -> list[sqlite3.Row]:
    uri = catalog_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT i.id AS image_id, i.content_sha256,
                   location.relative_path, result.subtitle_text
            FROM images i
            JOIN file_locations location ON location.id = (
                SELECT location2.id FROM file_locations location2
                WHERE location2.image_id=i.id AND location2.is_present=1
                ORDER BY location2.relative_path COLLATE NOCASE, location2.id
                LIMIT 1
            )
            JOIN analysis_results result ON result.id = (
                SELECT result2.id
                FROM analysis_results result2
                JOIN analysis_runs run2 ON run2.id=result2.run_id
                WHERE result2.image_id=i.id
                  AND run2.analysis_type='ocr'
                ORDER BY COALESCE(run2.completed_at, run2.started_at) DESC,
                         result2.id DESC
                LIMIT 1
            )
            WHERE trim(result.subtitle_text) <> ''
            ORDER BY i.content_sha256 ASC
            """
        ).fetchall()
    finally:
        connection.close()
    return rows


def _candidate(row: sqlite3.Row) -> SubtitleCandidate:
    return SubtitleCandidate(
        image_id=row["image_id"],
        relative_path=row["relative_path"],
        subtitle_text=row["subtitle_text"],
        source_text_sha256=hashlib.sha256(
            row["subtitle_text"].encode("utf-8")
        ).hexdigest(),
    )


def deterministic_subtitle_sample(
    catalog_path: Path, limit: int = MAX_TRIAL_IMAGES
) -> list[SubtitleCandidate]:
    if limit < 1 or limit > MAX_TRIAL_IMAGES:
        raise ValueError(f"sample limit must be between 1 and {MAX_TRIAL_IMAGES}")
    return [_candidate(row) for row in _latest_subtitle_rows(catalog_path)[:limit]]


def _top_level_group(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/")
    return normalized.split("/", 1)[0] if "/" in normalized else "(archive root)"


def representative_subtitle_sample(
    catalog_path: Path, limit: int = MAX_EVALUATION_IMAGES
) -> list[SubtitleCandidate]:
    """Proportional top-level stratification with stable content-hash ordering."""

    if limit < 1 or limit > MAX_EVALUATION_IMAGES:
        raise ValueError(f"sample limit must be between 1 and {MAX_EVALUATION_IMAGES}")
    rows = _latest_subtitle_rows(catalog_path)
    if len(rows) <= limit:
        return [_candidate(row) for row in rows]

    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[_top_level_group(row["relative_path"])].append(row)
    groups = sorted(grouped, key=str.casefold)
    if limit < len(groups):
        raise ValueError("sample limit is too small to represent every top-level group")

    quotas = {group: 1 for group in groups}
    remaining = limit - len(groups)
    exact_additions = {
        group: remaining * len(grouped[group]) / len(rows) for group in groups
    }
    for group in groups:
        addition = min(
            len(grouped[group]) - 1,
            math.floor(exact_additions[group]),
        )
        quotas[group] += addition
        remaining -= addition
    while remaining:
        eligible = [group for group in groups if quotas[group] < len(grouped[group])]
        if not eligible:
            break
        eligible.sort(
            key=lambda group: (
                -(exact_additions[group] - math.floor(exact_additions[group])),
                group.casefold(),
            )
        )
        for group in eligible:
            if not remaining:
                break
            quotas[group] += 1
            remaining -= 1

    selected: list[sqlite3.Row] = []
    for group in groups:
        selected.extend(grouped[group][: quotas[group]])
    selected.sort(
        key=lambda row: (row["content_sha256"], row["relative_path"].casefold())
    )
    return [_candidate(row) for row in selected]


def sample_distribution(candidates: Sequence[SubtitleCandidate]) -> dict[str, int]:
    counts = Counter(_top_level_group(item.relative_path) for item in candidates)
    return dict(sorted(counts.items(), key=lambda item: item[0].casefold()))


def evaluation_spec() -> SemanticSpec:
    base = SemanticSpec()
    parameters = dict(base.parameters)
    parameters.update(
        {
            "sampling_strategy": "top-level-proportional-min1-content-sha256-v1",
            "sample_group": "top-level relative-path component",
            "sample_allocation": "proportional-largest-remainder-with-minimum-one",
            "sample_order": "images.content_sha256 ASC within each group",
            "maximum_sample_size": MAX_EVALUATION_IMAGES,
        }
    )
    return replace(
        base,
        pipeline_version=EVALUATION_PIPELINE_VERSION,
        parameters=parameters,
    )


def model_shootout_specs() -> dict[str, SemanticSpec]:
    """Pinned, deployment-realistic models for the controlled 1,000-item run."""

    baseline = evaluation_spec()
    baseline_parameters = dict(baseline.parameters)
    baseline_parameters.update(
        {
            "license": "apache-2.0",
            "model_file": "model_optimized.onnx",
            "quantization": "dynamic-int8",
        }
    )
    baseline = replace(
        baseline,
        pipeline_version=SHOOTOUT_PIPELINE_VERSION,
        parameters=baseline_parameters,
    )

    e5_parameters = dict(baseline.parameters)
    e5_parameters.update(
        {
            "license": "mit",
            "maximum_tokens": 512,
            "query_prefix": "query: ",
            "document_prefix": "passage: ",
            "model_file": "onnx/model.onnx",
            "quantization": "float32",
            "fastembed_model_id": "image-finder/multilingual-e5-small-f32",
        }
    )
    e5 = replace(
        baseline,
        model_name="intfloat/multilingual-e5-small",
        model_source="intfloat/multilingual-e5-small",
        model_revision=E5_SMALL_REVISION,
        dimensions=384,
        parameters=e5_parameters,
    )

    mpnet_parameters = dict(baseline.parameters)
    mpnet_parameters.update(
        {
            "license": "apache-2.0",
            "maximum_tokens": 512,
            "query_prefix": None,
            "document_prefix": None,
            "model_file": "onnx/model_quantized.onnx",
            "quantization": "dynamic-int8",
            "fastembed_model_id": "image-finder/multilingual-mpnet-base-v2-int8",
        }
    )
    mpnet = replace(
        baseline,
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_source="Xenova/paraphrase-multilingual-mpnet-base-v2",
        model_revision=MPNET_REVISION,
        dimensions=768,
        parameters=mpnet_parameters,
    )
    return {"minilm": baseline, "e5-small": e5, "mpnet": mpnet}


def _manifest_hash(candidates: Sequence[SubtitleCandidate]) -> str:
    return _canonical_hash(
        [
            {
                "image_id": candidate.image_id,
                "relative_path": candidate.relative_path,
                "source_text_sha256": candidate.source_text_sha256,
            }
            for candidate in candidates
        ]
    )


def sample_manifest_hash(candidates: Sequence[SubtitleCandidate]) -> str:
    """Return the stable identity/path/source-text manifest hash for a sample."""

    return _manifest_hash(candidates)


def _normalized(values: Sequence[float], dimensions: int) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if len(vector) != dimensions:
        raise ValueError(f"expected {dimensions} dimensions, received {len(vector)}")
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        raise ValueError("embedding vector has zero magnitude")
    return tuple(value / norm for value in vector)


def _prefixed_texts(
    texts: Sequence[str], spec: SemanticSpec, parameter: str
) -> list[str]:
    prefix = spec.parameters.get(parameter)
    if prefix is None:
        return list(texts)
    if not isinstance(prefix, str):
        raise ValueError(f"{parameter} must be a string or null")
    return [prefix + text for text in texts]


def _connect_index(index_path: Path) -> sqlite3.Connection:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(index_path)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS semantic_runs (
            id TEXT PRIMARY KEY,
            spec_json TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            sample_size INTEGER NOT NULL,
            selection_seconds REAL NOT NULL,
            embedding_seconds REAL NOT NULL,
            built_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS semantic_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS subtitle_embeddings (
            run_id TEXT NOT NULL REFERENCES semantic_runs(id) ON DELETE CASCADE,
            image_id TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            subtitle_text TEXT NOT NULL,
            source_text_sha256 TEXT NOT NULL,
            vector_f32 BLOB NOT NULL,
            PRIMARY KEY (run_id, image_id)
        );
        """
    )
    connection.execute(
        "INSERT INTO semantic_metadata(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(INDEX_SCHEMA_VERSION),),
    )
    connection.commit()
    return connection


def build_trial_index(
    catalog_path: Path,
    index_path: Path,
    embedder: Embedder,
    spec: SemanticSpec = SemanticSpec(),
    limit: int = MAX_TRIAL_IMAGES,
    rebuild: bool = False,
) -> BuildSummary:
    total_started = time.perf_counter()
    selection_started = time.perf_counter()
    strategy = spec.parameters.get("sampling_strategy", "content-sha256-global-v1")
    if strategy == "top-level-proportional-min1-content-sha256-v1":
        candidates = representative_subtitle_sample(catalog_path, limit)
    else:
        candidates = deterministic_subtitle_sample(catalog_path, limit)
    selection_seconds = time.perf_counter() - selection_started
    manifest_sha256 = _manifest_hash(candidates)
    connection = _connect_index(index_path)
    try:
        existing = connection.execute(
            "SELECT * FROM semantic_runs WHERE id=? AND manifest_sha256=?",
            (spec.version_id, manifest_sha256),
        ).fetchone()
        count = connection.execute(
            "SELECT COUNT(*) FROM subtitle_embeddings WHERE run_id=?",
            (spec.version_id,),
        ).fetchone()[0]
        if not rebuild and existing is not None and count == len(candidates):
            return BuildSummary(
                run_id=spec.version_id,
                sample_size=len(candidates),
                manifest_sha256=manifest_sha256,
                selection_seconds=selection_seconds,
                embedding_seconds=float(existing["embedding_seconds"]),
                total_seconds=time.perf_counter() - total_started,
                reused=True,
            )

        embedding_started = time.perf_counter()
        document_texts = _prefixed_texts(
            [item.subtitle_text for item in candidates], spec, "document_prefix"
        )
        raw_vectors = list(embedder.embed(document_texts))
        embedding_seconds = time.perf_counter() - embedding_started
        if len(raw_vectors) != len(candidates):
            raise ValueError("embedder returned a different number of vectors than texts")
        vectors = [_normalized(vector, spec.dimensions) for vector in raw_vectors]

        connection.execute("DELETE FROM subtitle_embeddings")
        connection.execute("DELETE FROM semantic_runs")
        connection.execute(
            """
            INSERT INTO semantic_runs(
                id, spec_json, manifest_sha256, sample_size,
                selection_seconds, embedding_seconds
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                spec.version_id,
                json.dumps(spec.as_dict(), ensure_ascii=False, sort_keys=True),
                manifest_sha256,
                len(candidates),
                selection_seconds,
                embedding_seconds,
            ),
        )
        connection.executemany(
            """
            INSERT INTO subtitle_embeddings(
                run_id, image_id, relative_path, subtitle_text,
                source_text_sha256, vector_f32
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    spec.version_id,
                    item.image_id,
                    item.relative_path,
                    item.subtitle_text,
                    item.source_text_sha256,
                    struct.pack(f"<{spec.dimensions}f", *vector),
                )
                for item, vector in zip(candidates, vectors, strict=True)
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return BuildSummary(
        run_id=spec.version_id,
        sample_size=len(candidates),
        manifest_sha256=manifest_sha256,
        selection_seconds=selection_seconds,
        embedding_seconds=embedding_seconds,
        total_seconds=time.perf_counter() - total_started,
        reused=False,
    )


def semantic_search(
    index_path: Path,
    query: str,
    embedder: Embedder,
    spec: SemanticSpec = SemanticSpec(),
    limit: int = 5,
) -> list[SemanticResult]:
    text = query.strip()
    if not text or limit <= 0:
        return []
    query_text = _prefixed_texts([text], spec, "query_prefix")[0]
    query_vector = _normalized(embedder.embed([query_text])[0], spec.dimensions)
    connection = _connect_index(index_path)
    try:
        run = connection.execute(
            "SELECT spec_json FROM semantic_runs WHERE id=?", (spec.version_id,)
        ).fetchone()
        if run is None:
            raise RuntimeError("semantic index is missing or uses a different version")
        if json.loads(run["spec_json"]) != spec.as_dict():
            raise RuntimeError("semantic index metadata does not match the requested spec")
        rows = connection.execute(
            """
            SELECT image_id, relative_path, subtitle_text,
                   source_text_sha256, vector_f32
            FROM subtitle_embeddings WHERE run_id=?
            """,
            (spec.version_id,),
        ).fetchall()
    finally:
        connection.close()
    results = []
    for row in rows:
        vector = struct.unpack(f"<{spec.dimensions}f", row["vector_f32"])
        score = sum(left * right for left, right in zip(query_vector, vector))
        results.append(
            SemanticResult(
                image_id=row["image_id"],
                relative_path=row["relative_path"],
                subtitle_text=row["subtitle_text"],
                source_text_sha256=row["source_text_sha256"],
                score=score,
            )
        )
    return sorted(results, key=lambda item: (-item.score, item.relative_path.casefold()))[
        :limit
    ]


def indexed_lexical_search(
    index_path: Path,
    query: str,
    spec: SemanticSpec,
    limit: int = 20,
) -> list[SemanticResult]:
    """Exact/partial/OpenCC scoring over the evaluation index only."""

    variants = query_variants(query)
    if not variants or limit <= 0:
        return []
    connection = _connect_index(index_path)
    try:
        rows = connection.execute(
            """
            SELECT image_id, relative_path, subtitle_text, source_text_sha256
            FROM subtitle_embeddings WHERE run_id=?
            """,
            (spec.version_id,),
        ).fetchall()
    finally:
        connection.close()
    results: list[SemanticResult] = []
    for row in rows:
        text = row["subtitle_text"].casefold()
        path = row["relative_path"].casefold()
        scores: list[float] = []
        for variant in variants:
            needle = variant.casefold()
            if needle == text:
                scores.append(1.0)
            elif needle in text:
                ratio = min(1.0, len(needle) / max(1, len(text)))
                scores.append(0.8 + 0.2 * ratio)
            elif needle in path:
                scores.append(0.5)
        if scores:
            results.append(
                SemanticResult(
                    image_id=row["image_id"],
                    relative_path=row["relative_path"],
                    subtitle_text=row["subtitle_text"],
                    source_text_sha256=row["source_text_sha256"],
                    score=max(scores),
                )
            )
    return sorted(results, key=lambda item: (-item.score, item.relative_path.casefold()))[
        :limit
    ]


def hybrid_search(
    index_path: Path,
    query: str,
    embedder: Embedder,
    spec: SemanticSpec,
    limit: int = 10,
    rrf_k: int = 60,
) -> list[HybridResult]:
    """Equal-weight reciprocal-rank fusion over the sampled semantic index."""

    semantic = semantic_search(
        index_path, query, embedder, spec, limit=MAX_EVALUATION_IMAGES
    )
    lexical = indexed_lexical_search(
        index_path, query, spec, limit=MAX_EVALUATION_IMAGES
    )
    return fuse_rankings(semantic, lexical, limit=limit, rrf_k=rrf_k)


def fuse_rankings(
    semantic: Sequence[SemanticResult],
    lexical: Sequence[SemanticResult],
    limit: int = 10,
    rrf_k: int = 60,
) -> list[HybridResult]:
    """Fuse precomputed rankings without calibrating cosine as probability."""

    semantic_rank = {item.image_id: rank for rank, item in enumerate(semantic, 1)}
    lexical_rank = {item.image_id: rank for rank, item in enumerate(lexical, 1)}
    semantic_by_id = {item.image_id: item for item in semantic}
    lexical_by_id = {item.image_id: item for item in lexical}
    fused: list[HybridResult] = []
    for image_id in semantic_rank.keys() | lexical_rank.keys():
        item = semantic_by_id.get(image_id) or lexical_by_id[image_id]
        score = 0.0
        if image_id in semantic_rank:
            score += 1.0 / (rrf_k + semantic_rank[image_id])
        if image_id in lexical_rank:
            score += 1.0 / (rrf_k + lexical_rank[image_id])
        fused.append(
            HybridResult(
                image_id=image_id,
                relative_path=item.relative_path,
                subtitle_text=item.subtitle_text,
                source_text_sha256=item.source_text_sha256,
                hybrid_score=score,
                semantic_score=semantic_by_id[image_id].score,
                lexical_score=(
                    lexical_by_id[image_id].score if image_id in lexical_by_id else None
                ),
            )
        )
    return sorted(
        fused,
        key=lambda item: (-item.hybrid_score, item.relative_path.casefold()),
    )[:limit]
