from __future__ import annotations

import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from PIL import Image

from image_finder.catalog import Catalog
from image_finder.app_search import (
    AppSearchService,
    SemanticIndexUnavailable,
    load_semantic_presets,
)
from image_finder.semantic_search import (
    MAX_TRIAL_IMAGES,
    SemanticSpec,
    all_subtitle_candidates,
    application_spec,
    build_trial_index,
    deterministic_subtitle_sample,
    hybrid_search,
    indexed_lexical_search,
    model_shootout_specs,
    representative_subtitle_sample,
    sample_distribution,
    semantic_search,
    semantic_index_status,
)


class FakeEmbedder:
    def __init__(self, vectors: dict[str, Sequence[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls.append(tuple(texts))
        return [self.vectors[text] for text in texts]


def make_catalog(root: Path, subtitles: dict[str, str]) -> tuple[Path, Path]:
    project = root / "project"
    library = root / "library"
    project.mkdir()
    library.mkdir()
    for number, name in enumerate(subtitles):
        target = library / name
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 8), (number * 30, 10, 20)).save(target)
    database = project / "data" / "catalog.sqlite3"
    catalog = Catalog(database, project)
    try:
        catalog.scan_library(library)
        catalog.connection.execute(
            """
            INSERT INTO analysis_runs(
                id, analysis_type, engine_name, engine_version, model_name,
                model_version, pipeline_version, parameters_json,
                started_at, completed_at
            ) VALUES ('ocr-test', 'ocr', 'fake', '1', 'fake', '1', 'test',
                      '{}', '2026-01-01', '2026-01-01')
            """
        )
        rows = catalog.connection.execute(
            """
            SELECT i.id, fl.relative_path
            FROM images i JOIN file_locations fl ON fl.image_id=i.id
            WHERE fl.is_present=1
            """
        ).fetchall()
        for row in rows:
            text = subtitles[row["relative_path"]]
            catalog.connection.execute(
                """
                INSERT INTO analysis_results(
                    run_id, image_id, all_text, subtitle_text, confidence,
                    payload_json, created_at
                ) VALUES ('ocr-test', ?, ?, ?, 0.9, '{}', '2026-01-01')
                """,
                (row["id"], text, text),
            )
        catalog.connection.commit()
    finally:
        catalog.close()
    return project, database


def fake_spec(**changes: object) -> SemanticSpec:
    base = SemanticSpec(
        engine_name="fake",
        engine_version="1",
        runtime_name="fake-runtime",
        runtime_version="1",
        model_name="fake-model",
        model_source="local/fake-model",
        model_revision="revision-1",
        pipeline_version="test-v1",
        dimensions=2,
        parameters={
            "source_field": "ocr.subtitle_text",
            "sample_order": "images.content_sha256 ASC",
            "maximum_sample_size": MAX_TRIAL_IMAGES,
            "distance": "cosine",
            "vector_dtype": "float32",
            "l2_normalize": True,
        },
    )
    return replace(base, **changes)


class SemanticVersioningTests(unittest.TestCase):
    def test_version_changes_for_model_dimensions_pipeline_and_parameters(self) -> None:
        spec = fake_spec()
        self.assertNotEqual(spec.version_id, replace(spec, model_revision="revision-2").version_id)
        self.assertNotEqual(spec.version_id, replace(spec, dimensions=3).version_id)
        self.assertNotEqual(spec.version_id, replace(spec, pipeline_version="test-v2").version_id)
        changed_parameters = dict(spec.parameters, l2_normalize=False)
        self.assertNotEqual(
            spec.version_id,
            replace(spec, parameters=changed_parameters).version_id,
        )

    def test_shootout_specs_pin_models_and_prefix_behavior(self) -> None:
        specs = model_shootout_specs()
        self.assertEqual(set(specs), {"minilm", "e5-small", "mpnet"})
        self.assertEqual(specs["minilm"].dimensions, 384)
        self.assertEqual(specs["e5-small"].dimensions, 384)
        self.assertEqual(specs["mpnet"].dimensions, 768)
        self.assertTrue(
            all(spec.parameters["maximum_tokens"] == 512 for spec in specs.values())
        )
        self.assertEqual(specs["e5-small"].parameters["query_prefix"], "query: ")
        self.assertEqual(
            specs["e5-small"].parameters["document_prefix"], "passage: "
        )
        self.assertEqual(
            specs["mpnet"].parameters["model_file"], "onnx/model_quantized.onnx"
        )
        self.assertTrue(all(len(spec.model_revision) == 40 for spec in specs.values()))
        self.assertEqual(len({spec.version_id for spec in specs.values()}), 3)

    def test_build_reuses_then_rebuilds_for_force_or_source_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, catalog_path = make_catalog(
                root, {"one.png": "難過", "two.png": "開心"}
            )
            index_path = project / "data" / "semantic.sqlite3"
            embedder = FakeEmbedder({"難過": [1, 0], "開心": [0, 1], "非常難過": [1, 0]})
            spec = fake_spec()

            first = build_trial_index(catalog_path, index_path, embedder, spec)
            second = build_trial_index(catalog_path, index_path, embedder, spec)
            forced = build_trial_index(
                catalog_path, index_path, embedder, spec, rebuild=True
            )
            self.assertFalse(first.reused)
            self.assertTrue(second.reused)
            self.assertFalse(forced.reused)
            self.assertEqual(len(embedder.calls), 2)

            connection = sqlite3.connect(catalog_path)
            connection.execute(
                "UPDATE analysis_results SET subtitle_text='非常難過', all_text='非常難過' "
                "WHERE subtitle_text='難過'"
            )
            connection.commit()
            connection.close()
            changed = build_trial_index(catalog_path, index_path, embedder, spec)
            self.assertFalse(changed.reused)
            self.assertNotEqual(first.manifest_sha256, changed.manifest_sha256)
            self.assertEqual(len(embedder.calls), 3)


class SemanticSelectionAndRankingTests(unittest.TestCase):
    def test_representative_selection_is_deterministic_and_covers_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subtitles = {
                **{f"GroupAlpha/{number}.png": f"合成甲{number}" for number in range(6)},
                **{f"GroupBeta/{number}.png": f"合成乙{number}" for number in range(3)},
                "GroupGamma/only.png": "合成丙",
            }
            _, catalog_path = make_catalog(root, subtitles)
            first = representative_subtitle_sample(catalog_path, 6)
            second = representative_subtitle_sample(catalog_path, 6)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 6)
            distribution = sample_distribution(first)
            self.assertEqual(set(distribution), {"GroupAlpha", "GroupBeta", "GroupGamma"})
            self.assertGreater(distribution["GroupAlpha"], distribution["GroupGamma"])

    def test_selection_is_deterministic_bounded_and_uses_latest_ocr_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, catalog_path = make_catalog(
                root,
                {"c.png": "第三", "a.png": "第一", "b.png": "第二", "empty.png": ""},
            )
            connection = sqlite3.connect(catalog_path)
            connection.row_factory = sqlite3.Row
            stale_id = connection.execute(
                """
                SELECT i.id FROM images i
                JOIN file_locations fl ON fl.image_id=i.id
                WHERE fl.relative_path='c.png'
                """
            ).fetchone()["id"]
            connection.execute(
                """
                INSERT INTO analysis_runs(
                    id, analysis_type, engine_name, engine_version, model_name,
                    model_version, pipeline_version, parameters_json,
                    started_at, completed_at
                ) VALUES ('ocr-new', 'ocr', 'fake', '2', 'fake', '2', 'test-v2',
                          '{}', '2026-02-01', '2026-02-01')
                """
            )
            connection.execute(
                """
                INSERT INTO analysis_results(
                    run_id, image_id, all_text, subtitle_text, confidence,
                    payload_json, created_at
                ) VALUES ('ocr-new', ?, '', '', 0.9, '{}', '2026-02-01')
                """,
                (stale_id,),
            )
            connection.commit()
            connection.close()

            first = deterministic_subtitle_sample(catalog_path, 2)
            second = deterministic_subtitle_sample(catalog_path, 2)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 2)
            all_current = deterministic_subtitle_sample(catalog_path, MAX_TRIAL_IMAGES)
            self.assertNotIn(stale_id, {item.image_id for item in all_current})

            connection = sqlite3.connect(catalog_path)
            connection.row_factory = sqlite3.Row
            expected = [
                row["id"]
                for row in connection.execute(
                    """
                    SELECT i.id
                    FROM images i
                    JOIN analysis_results ar ON ar.id = (
                        SELECT ar2.id FROM analysis_results ar2
                        JOIN analysis_runs run2 ON run2.id=ar2.run_id
                        WHERE ar2.image_id=i.id AND run2.analysis_type='ocr'
                        ORDER BY COALESCE(run2.completed_at, run2.started_at) DESC,
                                 ar2.id DESC LIMIT 1
                    )
                    WHERE trim(ar.subtitle_text) <> ''
                    ORDER BY i.content_sha256 LIMIT 2
                    """
                )
            ]
            connection.close()
            self.assertEqual([item.image_id for item in first], expected)
            with self.assertRaises(ValueError):
                deterministic_subtitle_sample(catalog_path, MAX_TRIAL_IMAGES + 1)

    def test_cosine_ranking_returns_closest_subtitle_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, catalog_path = make_catalog(
                root, {"sad.png": "我真的很難過", "happy.png": "今天太開心了"}
            )
            index_path = project / "data" / "semantic.sqlite3"
            embedder = FakeEmbedder(
                {
                    "我真的很難過": [1, 0],
                    "今天太開心了": [0, 1],
                    "沮喪的感覺": [0.9, 0.1],
                }
            )
            spec = fake_spec()
            build_trial_index(catalog_path, index_path, embedder, spec)
            results = semantic_search(index_path, "沮喪的感覺", embedder, spec)
            self.assertEqual(results[0].relative_path, "sad.png")
            self.assertGreater(results[0].score, results[1].score)
            self.assertEqual(len(results[0].source_text_sha256), 64)

    def test_model_prefixes_are_applied_only_to_embedding_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, catalog_path = make_catalog(root, {"sad.png": "難過"})
            index_path = project / "data" / "semantic.sqlite3"
            embedder = FakeEmbedder(
                {"passage: 難過": [1, 0], "query: 沮喪": [1, 0]}
            )
            base = fake_spec()
            spec = replace(
                base,
                parameters=dict(
                    base.parameters,
                    document_prefix="passage: ",
                    query_prefix="query: ",
                ),
            )
            build_trial_index(catalog_path, index_path, embedder, spec)
            results = semantic_search(index_path, "沮喪", embedder, spec)
            self.assertEqual(embedder.calls, [("passage: 難過",), ("query: 沮喪",)])
            self.assertEqual(results[0].subtitle_text, "難過")

    def test_hybrid_rrf_boosts_lexical_match_without_replacing_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, catalog_path = make_catalog(
                root,
                {"sad.png": "我很難過", "happy.png": "今天太開心了"},
            )
            index_path = project / "data" / "semantic.sqlite3"
            embedder = FakeEmbedder(
                {
                    "我很難過": [1, 0],
                    "今天太開心了": [0, 1],
                    "難過": [0.1, 0.9],
                }
            )
            spec = fake_spec()
            build_trial_index(catalog_path, index_path, embedder, spec)
            semantic = semantic_search(index_path, "難過", embedder, spec)
            lexical = indexed_lexical_search(index_path, "難過", spec)
            hybrid = hybrid_search(index_path, "難過", embedder, spec)
            self.assertEqual(semantic[0].relative_path, "happy.png")
            self.assertEqual(lexical[0].relative_path, "sad.png")
            self.assertEqual(hybrid[0].relative_path, "sad.png")
            self.assertIsNotNone(hybrid[0].lexical_score)


class ExistingTextSearchRegressionTests(unittest.TestCase):
    def test_exact_and_partial_catalog_search_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, catalog_path = make_catalog(
                root,
                {"alpha.png": "合成測試冰品字幕", "beta.png": "这是测试教训吗"},
            )
            catalog = Catalog(catalog_path, project)
            try:
                self.assertEqual(
                    [item.relative_path for item in catalog.search("合成測試冰品字幕")],
                    ["alpha.png"],
                )
                self.assertEqual(
                    [item.relative_path for item in catalog.search("冰品")],
                    ["alpha.png"],
                )
                self.assertEqual(
                    [item.relative_path for item in catalog.search("教訓")],
                    ["beta.png"],
                )
            finally:
                catalog.close()


class ApplicationSearchTests(unittest.TestCase):
    @staticmethod
    def application_fake_spec() -> SemanticSpec:
        base = fake_spec()
        return replace(
            base,
            pipeline_version="synthetic-application-v1",
            parameters=dict(
                base.parameters,
                sampling_strategy="all-current-subtitles-content-sha256-v1",
                maximum_sample_size="all eligible current OCR subtitles",
            ),
        )

    def test_application_spec_pins_full_catalog_minilm(self) -> None:
        spec = application_spec()
        self.assertEqual(spec.dimensions, 384)
        self.assertEqual(
            spec.parameters["sampling_strategy"],
            "all-current-subtitles-content-sha256-v1",
        )
        self.assertEqual(
            spec.parameters["maximum_sample_size"],
            "all eligible current OCR subtitles",
        )
        self.assertEqual(spec.parameters["batch_size"], 32)

    def test_full_application_index_status_detects_missing_ready_and_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, catalog_path = make_catalog(
                root,
                {"one.png": "合成第一句", "two.png": "合成第二句"},
            )
            index_path = project / "data" / "application-semantic.sqlite3"
            spec = self.application_fake_spec()
            missing = semantic_index_status(catalog_path, index_path, spec)
            self.assertFalse(missing.ready)
            self.assertEqual(missing.reason, "missing")
            self.assertEqual(len(all_subtitle_candidates(catalog_path)), 2)

            embedder = FakeEmbedder(
                {"合成第一句": [1, 0], "合成第二句": [0, 1]}
            )
            build_trial_index(catalog_path, index_path, embedder, spec)
            ready = semantic_index_status(catalog_path, index_path, spec)
            self.assertTrue(ready.ready)
            self.assertEqual(ready.indexed_count, 2)

            connection = sqlite3.connect(catalog_path)
            connection.execute(
                "UPDATE analysis_results SET subtitle_text='合成已改變', "
                "all_text='合成已改變' WHERE subtitle_text='合成第一句'"
            )
            connection.commit()
            connection.close()
            stale = semantic_index_status(catalog_path, index_path, spec)
            self.assertFalse(stale.ready)
            self.assertEqual(stale.reason, "source_changed")

    def test_literal_semantic_presets_and_more_keep_rank_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, catalog_path = make_catalog(
                root,
                {
                    "reaction.png": "合成驚訝字幕",
                    "middle.png": "合成普通字幕",
                    "other.png": "合成其他字幕",
                },
            )
            source_files = list((root / "library").glob("*.png"))
            source_bytes = {path: path.read_bytes() for path in source_files}
            index_path = project / "data" / "application-semantic.sqlite3"
            spec = self.application_fake_spec()
            embedder = FakeEmbedder(
                {
                    "合成驚訝字幕": [1, 0],
                    "合成普通字幕": [0.7, 0.3],
                    "合成其他字幕": [0, 1],
                    "驚訝反應": [1, 0],
                }
            )
            build_trial_index(catalog_path, index_path, embedder, spec)
            catalog = Catalog(catalog_path, project)
            try:
                service = AppSearchService(
                    catalog,
                    index_path,
                    project / "models",
                    spec=spec,
                    embedder_factory=lambda: embedder,
                )
                literal = service.search("普通", "literal")
                self.assertEqual(
                    [match.result.relative_path for match in literal.matches],
                    ["middle.png"],
                )

                first = service.search(
                    "驚訝反應", "semantic", visible_limit=2
                )
                expanded = service.search(
                    "驚訝反應", "semantic", visible_limit=3
                )
                self.assertTrue(first.has_more)
                self.assertFalse(expanded.has_more)
                self.assertEqual(
                    [match.result.image_id for match in first.matches],
                    [match.result.image_id for match in expanded.matches[:2]],
                )
                self.assertEqual(
                    expanded.matches[0].result.relative_path, "reaction.png"
                )
                for path, payload in source_bytes.items():
                    self.assertEqual(path.read_bytes(), payload)
            finally:
                catalog.close()

            presets = load_semantic_presets(
                Path(__file__).resolve().parents[1] / "semantic-presets.json"
            )
            self.assertEqual(
                {preset.label: preset.query for preset in presets}["無奈"],
                "很無奈",
            )

    def test_missing_semantic_index_is_reported_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, catalog_path = make_catalog(root, {"one.png": "合成字幕"})
            catalog = Catalog(catalog_path, project)
            try:
                service = AppSearchService(
                    catalog,
                    project / "data" / "missing.sqlite3",
                    project / "models",
                    spec=self.application_fake_spec(),
                    embedder_factory=lambda: FakeEmbedder({}),
                )
                with self.assertRaises(SemanticIndexUnavailable) as raised:
                    service.search("合成概念", "semantic")
                self.assertEqual(raised.exception.status.reason, "missing")
            finally:
                catalog.close()


if __name__ == "__main__":
    unittest.main()
