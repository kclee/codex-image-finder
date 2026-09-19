"""Bounded, rebuildable visual-semantic indexing for reviewed images only."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image, ImageOps


VISUAL_INDEX_SCHEMA_VERSION = 1
VISUAL_PIPELINE_VERSION = "reviewed-images-siglip2-visual-v1"
VISUAL_MODEL_NAME = "google/siglip2-base-patch16-224"
VISUAL_MODEL_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
VISUAL_ONNX_SOURCE = "onnx-community/siglip2-base-patch16-224-ONNX"
VISUAL_ONNX_REVISION = "ba1f3b0843f24bc5417d38e19c37b287d719b2f4"
VISUAL_DIMENSIONS = 768


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class VisualSpec:
    engine_name: str = "ONNX Runtime"
    engine_version: str = "1.29.0"
    tokenizer_name: str = "Hugging Face Tokenizers"
    tokenizer_version: str = "0.23.2"
    pillow_version: str = "12.3.0"
    numpy_version: str = "2.3.5"
    model_name: str = VISUAL_MODEL_NAME
    model_revision: str = VISUAL_MODEL_REVISION
    onnx_source: str = VISUAL_ONNX_SOURCE
    onnx_revision: str = VISUAL_ONNX_REVISION
    pipeline_version: str = VISUAL_PIPELINE_VERSION
    dimensions: int = VISUAL_DIMENSIONS
    index_schema_version: int = VISUAL_INDEX_SCHEMA_VERSION
    parameters: dict[str, object] = field(
        default_factory=lambda: {
            "vision_model_file": "onnx/vision_model_int8.onnx",
            "vision_model_sha256": (
                "0dd31785a2713f1113ef2272472165c69d580473dae38d7b47568ac587795e70"
            ),
            "text_model_file": "onnx/text_model_int8.onnx",
            "text_model_sha256": (
                "3a0603d3a00c05a80a6ded4743c16aaac7b1e62cdcc7e362e7ce418659b96400"
            ),
            "quantization": "int8",
            "image_mode": "RGB",
            "exif_transpose": True,
            "resize": [224, 224],
            "resize_behavior": "direct fixed-size resize; aspect ratio not preserved",
            "resample": "Pillow BICUBIC",
            "rescale_factor": 1 / 255,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "channel_order": "RGB then NCHW",
            "text_max_tokens": 64,
            "text_padding": "right/max_length",
            "text_add_eos": True,
            "l2_normalize": True,
            "distance": "cosine",
            "vector_dtype": "float32 little-endian",
            "selection": "unique content hashes present in completed human review",
        }
    )

    @property
    def version_id(self) -> str:
        return _canonical_hash(self.as_dict())[:24]

    def as_dict(self) -> dict[str, object]:
        return {
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_version": self.tokenizer_version,
            "pillow_version": self.pillow_version,
            "numpy_version": self.numpy_version,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "onnx_source": self.onnx_source,
            "onnx_revision": self.onnx_revision,
            "pipeline_version": self.pipeline_version,
            "dimensions": self.dimensions,
            "index_schema_version": self.index_schema_version,
            "parameters": self.parameters,
        }


@dataclass(frozen=True, slots=True)
class VisualCandidate:
    image_id: str
    content_sha256: str
    relative_path: str
    source_path: Path
    subtitle_text: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class VisualBuildSummary:
    run_id: str
    image_count: int
    manifest_sha256: str
    embedding_seconds: float
    total_seconds: float
    reused: bool


@dataclass(frozen=True, slots=True)
class VisualResult:
    image_id: str
    content_sha256: str
    relative_path: str
    subtitle_text: str
    score: float


class VisualEmbedder(Protocol):
    def embed_images(self, paths: Sequence[Path]) -> Sequence[Sequence[float]]: ...

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


def review_pair_labels(review: dict[str, object]) -> dict[tuple[str, str], str]:
    """Collapse repeated reviewed rows only when their human labels agree."""
    grouped: dict[tuple[str, str], set[str]] = {}
    for item in review["ratings"]:
        key = (str(item["query"]), str(item["image_id"]))
        grouped.setdefault(key, set()).add(str(item["human_rating"]))
    return {
        key: next(iter(values)) if len(values) == 1 else "conflict"
        for key, values in grouped.items()
    }


def reviewed_image_candidates(
    review: dict[str, object], catalog_path: Path
) -> list[VisualCandidate]:
    """Resolve only reviewed image identities through the read-only catalog."""
    identities: dict[str, tuple[str, str]] = {}
    for item in review["ratings"]:
        content_hash = str(item["image_content_sha256"])
        pair = (str(item["image_id"]), content_hash)
        previous = identities.get(content_hash)
        if previous is not None and previous != pair:
            raise ValueError("one content hash maps to multiple reviewed image identities")
        identities[content_hash] = pair

    connection = sqlite3.connect(catalog_path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        candidates = []
        for content_hash, (image_id, expected_hash) in sorted(identities.items()):
            row = connection.execute(
                """
                SELECT i.id, i.content_sha256, i.width, i.height,
                       fl.relative_path, fl.observed_full_path,
                       COALESCE(ar.subtitle_text, '') AS subtitle_text
                FROM images i
                JOIN file_locations fl ON fl.image_id=i.id AND fl.is_present=1
                LEFT JOIN analysis_results ar ON ar.id=(
                    SELECT ar2.id FROM analysis_results ar2
                    JOIN analysis_runs run2 ON run2.id=ar2.run_id
                    WHERE ar2.image_id=i.id AND run2.analysis_type='ocr'
                    ORDER BY COALESCE(run2.completed_at, run2.started_at) DESC,
                             ar2.id DESC LIMIT 1
                )
                WHERE i.id=?
                ORDER BY fl.id DESC LIMIT 1
                """,
                (image_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"reviewed image is absent from the current catalog: {image_id}")
            if row["content_sha256"] != expected_hash:
                raise ValueError(f"review/catalog content hash mismatch for {image_id}")
            source_path = Path(row["observed_full_path"])
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            candidates.append(
                VisualCandidate(
                    image_id=row["id"],
                    content_sha256=row["content_sha256"],
                    relative_path=row["relative_path"],
                    source_path=source_path,
                    subtitle_text=row["subtitle_text"],
                    width=int(row["width"]),
                    height=int(row["height"]),
                )
            )
        return candidates
    finally:
        connection.close()


def visual_manifest_hash(candidates: Sequence[VisualCandidate]) -> str:
    return _canonical_hash(
        [
            {"image_id": item.image_id, "content_sha256": item.content_sha256}
            for item in sorted(candidates, key=lambda value: value.content_sha256)
        ]
    )


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("model returned a zero-length embedding")
    return array / norms


class Siglip2OnnxEmbedder:
    """Minimal int8 ONNX adapter with explicit, portable preprocessing."""

    def __init__(
        self, model_dir: Path, spec: VisualSpec, image_batch_size: int = 8
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        installed = {
            "onnxruntime": importlib.metadata.version("onnxruntime"),
            "tokenizers": importlib.metadata.version("tokenizers"),
            "Pillow": importlib.metadata.version("Pillow"),
            "numpy": importlib.metadata.version("numpy"),
        }
        expected = {
            "onnxruntime": spec.engine_version,
            "tokenizers": spec.tokenizer_version,
            "Pillow": spec.pillow_version,
            "numpy": spec.numpy_version,
        }
        if installed != expected:
            raise RuntimeError(
                f"visual dependency mismatch: expected {expected}, found {installed}"
            )
        self.spec = spec
        self.model_dir = model_dir
        self.image_batch_size = image_batch_size
        vision_path = model_dir / str(spec.parameters["vision_model_file"])
        text_path = model_dir / str(spec.parameters["text_model_file"])
        for path, expected_hash in (
            (vision_path, spec.parameters["vision_model_sha256"]),
            (text_path, spec.parameters["text_model_sha256"]),
        ):
            actual = _file_sha256(path)
            if actual != expected_hash:
                raise RuntimeError(f"model file hash mismatch: {path}")
        self.vision = ort.InferenceSession(
            str(vision_path), providers=["CPUExecutionProvider"]
        )
        self.text = ort.InferenceSession(
            str(text_path), providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    @staticmethod
    def _preprocess_image(path: Path) -> np.ndarray:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image = image.resize((224, 224), Image.Resampling.BICUBIC)
            array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - 0.5) / 0.5
        return np.transpose(array, (2, 0, 1))

    def embed_images(self, paths: Sequence[Path]) -> np.ndarray:
        if self.vision is None:
            raise RuntimeError("the visual model session has been released")
        outputs = []
        for start in range(0, len(paths), self.image_batch_size):
            batch_paths = paths[start : start + self.image_batch_size]
            pixels = np.stack(
                [self._preprocess_image(path) for path in batch_paths]
            ).astype(np.float32, copy=False)
            pooled = self.vision.run(["pooler_output"], {"pixel_values": pixels})[0]
            outputs.append(np.asarray(pooled, dtype=np.float32))
        if not outputs:
            return np.empty((0, self.spec.dimensions), dtype=np.float32)
        normalized = _l2_normalize(np.concatenate(outputs, axis=0))
        if normalized.shape[1] != self.spec.dimensions:
            raise ValueError(f"unexpected image embedding shape: {normalized.shape}")
        return normalized

    def release_image_model(self) -> None:
        """Release the one-time indexing graph before serving text queries."""
        self.vision = None
        gc.collect()

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(list(texts))
        input_ids = np.asarray([item.ids for item in encodings], dtype=np.int64)
        if input_ids.shape[1] != int(self.spec.parameters["text_max_tokens"]):
            raise ValueError(f"unexpected tokenized shape: {input_ids.shape}")
        pooled = self.text.run(["pooler_output"], {"input_ids": input_ids})[0]
        normalized = _l2_normalize(np.asarray(pooled, dtype=np.float32))
        if normalized.shape[1] != self.spec.dimensions:
            raise ValueError(f"unexpected text embedding shape: {normalized.shape}")
        return normalized


def _connect_visual_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS visual_runs (
            run_id TEXT PRIMARY KEY,
            spec_json TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            image_count INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS visual_embeddings (
            run_id TEXT NOT NULL REFERENCES visual_runs(run_id) ON DELETE CASCADE,
            image_id TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            subtitle_text TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            vector_f32le BLOB NOT NULL,
            PRIMARY KEY (run_id, image_id)
        );
        CREATE TABLE IF NOT EXISTS visual_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    return connection


def build_visual_index(
    index_path: Path,
    candidates: Sequence[VisualCandidate],
    embedder: VisualEmbedder,
    spec: VisualSpec,
    rebuild: bool = False,
) -> VisualBuildSummary:
    started = time.perf_counter()
    ordered = sorted(candidates, key=lambda item: item.content_sha256)
    manifest = visual_manifest_hash(ordered)
    run_id = _canonical_hash({"spec": spec.as_dict(), "manifest": manifest})[:24]
    connection = _connect_visual_index(index_path)
    try:
        existing = connection.execute(
            "SELECT image_count FROM visual_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        stored_count = connection.execute(
            "SELECT COUNT(*) FROM visual_embeddings WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        if not rebuild and existing and stored_count == len(ordered) == existing[0]:
            return VisualBuildSummary(
                run_id=run_id,
                image_count=len(ordered),
                manifest_sha256=manifest,
                embedding_seconds=0.0,
                total_seconds=time.perf_counter() - started,
                reused=True,
            )

        embedding_started = time.perf_counter()
        vectors = np.asarray(
            embedder.embed_images([item.source_path for item in ordered]),
            dtype=np.float32,
        )
        embedding_seconds = time.perf_counter() - embedding_started
        if vectors.shape != (len(ordered), spec.dimensions):
            raise ValueError(f"unexpected visual embedding matrix: {vectors.shape}")

        connection.execute("DELETE FROM visual_embeddings")
        connection.execute("DELETE FROM visual_runs")
        connection.execute(
            "INSERT INTO visual_runs(run_id,spec_json,manifest_sha256,image_count) "
            "VALUES(?,?,?,?)",
            (
                run_id,
                json.dumps(spec.as_dict(), ensure_ascii=False, sort_keys=True),
                manifest,
                len(ordered),
            ),
        )
        connection.executemany(
            """
            INSERT INTO visual_embeddings(
                run_id,image_id,content_sha256,relative_path,subtitle_text,
                width,height,vector_f32le
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            [
                (
                    run_id,
                    item.image_id,
                    item.content_sha256,
                    item.relative_path,
                    item.subtitle_text,
                    item.width,
                    item.height,
                    np.asarray(vector, dtype="<f4").tobytes(),
                )
                for item, vector in zip(ordered, vectors)
            ],
        )
        connection.execute(
            "INSERT INTO visual_metadata(key,value) VALUES('active_run_id',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (run_id,),
        )
        connection.commit()
        return VisualBuildSummary(
            run_id=run_id,
            image_count=len(ordered),
            manifest_sha256=manifest,
            embedding_seconds=embedding_seconds,
            total_seconds=time.perf_counter() - started,
            reused=False,
        )
    finally:
        connection.close()


def rank_visual_index(
    index_path: Path,
    query_vector: Sequence[float],
    spec: VisualSpec,
    limit: int,
) -> list[VisualResult]:
    connection = sqlite3.connect(index_path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        active = connection.execute(
            "SELECT value FROM visual_metadata WHERE key='active_run_id'"
        ).fetchone()
        if active is None:
            raise RuntimeError("visual index has no active run")
        run = connection.execute(
            "SELECT spec_json FROM visual_runs WHERE run_id=?", (active[0],)
        ).fetchone()
        if run is None or json.loads(run[0]) != spec.as_dict():
            raise RuntimeError("visual index version does not match the requested model")
        rows = connection.execute(
            """
            SELECT image_id,content_sha256,relative_path,subtitle_text,vector_f32le
            FROM visual_embeddings WHERE run_id=?
            """,
            (active[0],),
        ).fetchall()
    finally:
        connection.close()

    query = _l2_normalize(np.asarray(query_vector, dtype=np.float32))[0]
    scored = []
    for row in rows:
        vector = np.frombuffer(row["vector_f32le"], dtype="<f4")
        if vector.size != spec.dimensions:
            raise RuntimeError("stored visual vector has unexpected dimensions")
        scored.append(
            VisualResult(
                image_id=row["image_id"],
                content_sha256=row["content_sha256"],
                relative_path=row["relative_path"],
                subtitle_text=row["subtitle_text"],
                score=float(np.dot(query, vector)),
            )
        )
    return sorted(scored, key=lambda item: (-item.score, item.content_sha256))[:limit]


def visual_search(
    index_path: Path,
    query: str,
    embedder: VisualEmbedder,
    spec: VisualSpec,
    limit: int,
) -> list[VisualResult]:
    vector = embedder.embed_texts([query])[0]
    return rank_visual_index(index_path, vector, spec, limit)
