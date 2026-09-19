from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from image_finder.visual_semantic import (
    VisualCandidate,
    VisualSpec,
    build_visual_index,
    review_pair_labels,
    visual_manifest_hash,
    visual_search,
)


class FakeVisualEmbedder:
    def __init__(
        self,
        image_vectors: dict[str, Sequence[float]],
        text_vectors: dict[str, Sequence[float]],
    ) -> None:
        self.image_vectors = image_vectors
        self.text_vectors = text_vectors
        self.image_calls: list[tuple[str, ...]] = []
        self.text_calls: list[tuple[str, ...]] = []

    def embed_images(self, paths: Sequence[Path]) -> Sequence[Sequence[float]]:
        names = tuple(path.name for path in paths)
        self.image_calls.append(names)
        return [self.image_vectors[name] for name in names]

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        values = tuple(texts)
        self.text_calls.append(values)
        return [self.text_vectors[text] for text in values]


def fake_spec(**changes: object) -> VisualSpec:
    base = VisualSpec(
        engine_name="fake-runtime",
        engine_version="1",
        tokenizer_name="fake-tokenizer",
        tokenizer_version="1",
        pillow_version="1",
        numpy_version="1",
        model_name="synthetic-visual-model",
        model_revision="revision-1",
        onnx_source="local/synthetic",
        onnx_revision="onnx-revision-1",
        pipeline_version="synthetic-visual-v1",
        dimensions=2,
        parameters={
            "resize": [2, 2],
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "distance": "cosine",
            "vector_dtype": "float32 little-endian",
        },
    )
    return replace(base, **changes)


def make_candidate(root: Path, name: str, payload: bytes) -> VisualCandidate:
    path = root / name
    path.write_bytes(payload)
    content_hash = hashlib.sha256(payload).hexdigest()
    return VisualCandidate(
        image_id=f"synthetic-{name}",
        content_sha256=content_hash,
        relative_path=f"synthetic/{name}",
        source_path=path,
        subtitle_text=f"Synthetic subtitle for {name}",
        width=2,
        height=2,
    )


class VisualVersioningAndRebuildTests(unittest.TestCase):
    def test_version_changes_for_model_dimensions_pipeline_and_preprocessing(self) -> None:
        spec = fake_spec()
        self.assertNotEqual(
            spec.version_id, replace(spec, model_revision="revision-2").version_id
        )
        self.assertNotEqual(spec.version_id, replace(spec, dimensions=3).version_id)
        self.assertNotEqual(
            spec.version_id, replace(spec, pillow_version="2").version_id
        )
        self.assertNotEqual(
            spec.version_id,
            replace(spec, pipeline_version="synthetic-visual-v2").version_id,
        )
        self.assertNotEqual(
            spec.version_id,
            replace(spec, parameters=dict(spec.parameters, resize=[4, 4])).version_id,
        )

    def test_manifest_is_deterministic_and_changes_with_content_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = make_candidate(root, "one.bin", b"one")
            second = make_candidate(root, "two.bin", b"two")
            self.assertEqual(
                visual_manifest_hash([first, second]),
                visual_manifest_hash([second, first]),
            )
            changed = replace(first, content_sha256=hashlib.sha256(b"changed").hexdigest())
            self.assertNotEqual(
                visual_manifest_hash([first, second]),
                visual_manifest_hash([changed, second]),
            )

    def test_build_reuses_and_rebuilds_for_force_manifest_or_spec_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one = make_candidate(root, "one.bin", b"one")
            two = make_candidate(root, "two.bin", b"two")
            embedder = FakeVisualEmbedder(
                {"one.bin": [1, 0], "two.bin": [0, 1]}, {"query": [1, 0]}
            )
            index = root / "visual.sqlite3"
            spec = fake_spec()

            first = build_visual_index(index, [one, two], embedder, spec)
            reused = build_visual_index(index, [two, one], embedder, spec)
            forced = build_visual_index(index, [one, two], embedder, spec, rebuild=True)
            changed_manifest = build_visual_index(index, [one], embedder, spec)
            changed_spec = build_visual_index(
                index,
                [one],
                embedder,
                replace(spec, pipeline_version="synthetic-visual-v2"),
            )

            self.assertFalse(first.reused)
            self.assertTrue(reused.reused)
            self.assertFalse(forced.reused)
            self.assertFalse(changed_manifest.reused)
            self.assertFalse(changed_spec.reused)
            self.assertEqual(len(embedder.image_calls), 4)
            self.assertNotEqual(first.manifest_sha256, changed_manifest.manifest_sha256)

            connection = sqlite3.connect(index)
            run_count = connection.execute("SELECT COUNT(*) FROM visual_runs").fetchone()[0]
            row_count = connection.execute(
                "SELECT COUNT(*) FROM visual_embeddings"
            ).fetchone()[0]
            connection.close()
            self.assertEqual(run_count, 1)
            self.assertEqual(row_count, 1)


class VisualRankingAndReviewLabelTests(unittest.TestCase):
    def test_cosine_ranking_returns_closest_image_first_without_source_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reaction = make_candidate(root, "reaction.bin", b"reaction source")
            neutral = make_candidate(root, "neutral.bin", b"neutral source")
            original_bytes = {
                item.source_path: item.source_path.read_bytes()
                for item in (reaction, neutral)
            }
            embedder = FakeVisualEmbedder(
                {"reaction.bin": [1, 0], "neutral.bin": [0, 1]},
                {"awkward reaction": [0.9, 0.1]},
            )
            index = root / "visual.sqlite3"
            spec = fake_spec()
            build_visual_index(index, [neutral, reaction], embedder, spec)

            results = visual_search(
                index, "awkward reaction", embedder, spec, limit=2
            )

            self.assertEqual(results[0].relative_path, "synthetic/reaction.bin")
            self.assertGreater(results[0].score, results[1].score)
            self.assertEqual(embedder.text_calls, [("awkward reaction",)])
            for path, payload in original_bytes.items():
                self.assertEqual(path.read_bytes(), payload)

    def test_review_labels_keep_unreviewed_absent_and_mark_conflicts(self) -> None:
        review = {
            "ratings": [
                {"query": "q1", "image_id": "a", "human_rating": "useful"},
                {"query": "q1", "image_id": "a", "human_rating": "useful"},
                {"query": "q1", "image_id": "b", "human_rating": "maybe"},
                {"query": "q2", "image_id": "a", "human_rating": "useful"},
                {"query": "q2", "image_id": "a", "human_rating": "not_useful"},
            ]
        }
        labels = review_pair_labels(review)
        self.assertEqual(labels[("q1", "a")], "useful")
        self.assertEqual(labels[("q1", "b")], "maybe")
        self.assertEqual(labels[("q2", "a")], "conflict")
        self.assertNotIn(("q2", "b"), labels)


if __name__ == "__main__":
    unittest.main()
