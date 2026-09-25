"""UI-independent search orchestration for the first usable desktop application."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

from .catalog import Catalog
from .domain import SearchResult
from .semantic_search import (
    BuildSummary,
    Embedder,
    FastEmbedder,
    SemanticIndexStatus,
    SemanticSpec,
    application_spec,
    build_trial_index,
    semantic_index_status,
    semantic_neighbors,
    semantic_search,
)


SearchMode = Literal["literal", "semantic", "similar"]
SortOrder = Literal["relevance", "newest", "oldest"]
SEMANTIC_PAGE_SIZE = 24
SEMANTIC_DATE_CANDIDATE_LIMIT = 240
SEMANTIC_INDEX_FILENAME = "semantic-subtitle-v0.1.sqlite3"
MEASURED_MINILM_SUBTITLES_PER_SECOND = 6.53


class SemanticIndexUnavailable(RuntimeError):
    def __init__(self, status: SemanticIndexStatus) -> None:
        super().__init__(f"semantic index is not ready: {status.reason}")
        self.status = status


class SemanticSourceUnavailable(RuntimeError):
    def __init__(self, image_id: str) -> None:
        super().__init__("selected image has no subtitle embedding")
        self.image_id = image_id


@dataclass(frozen=True, slots=True)
class SemanticPreset:
    label: str
    query: str


@dataclass(frozen=True, slots=True)
class SearchMatch:
    result: SearchResult
    semantic_score: float | None = None


@dataclass(frozen=True, slots=True)
class SearchPage:
    mode: SearchMode
    query: str
    matches: tuple[SearchMatch, ...]
    total: int
    visible_limit: int

    @property
    def has_more(self) -> bool:
        return len(self.matches) < self.total


def load_semantic_presets(path: Path) -> list[SemanticPreset]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("presets"), list):
        raise ValueError("unsupported semantic preset configuration")
    presets = []
    for item in payload["presets"]:
        label = str(item.get("label", "")).strip()
        query = str(item.get("query", "")).strip()
        if not label or not query:
            raise ValueError("semantic preset labels and queries must not be empty")
        presets.append(SemanticPreset(label=label, query=query))
    if not presets:
        raise ValueError("semantic preset configuration is empty")
    return presets


def estimated_semantic_build_seconds(eligible_count: int) -> float:
    return eligible_count / MEASURED_MINILM_SUBTITLES_PER_SECOND


class AppSearchService:
    """Keep literal and meaning search explicit while sharing catalog result records."""

    def __init__(
        self,
        catalog: Catalog,
        index_path: Path,
        model_cache: Path,
        spec: SemanticSpec | None = None,
        embedder_factory: Callable[[], Embedder] | None = None,
    ) -> None:
        self.catalog = catalog
        self.index_path = index_path
        self.model_cache = model_cache
        self.spec = spec or application_spec()
        self._embedder_factory = embedder_factory
        self._embedder: Embedder | None = None

    def index_status(self) -> SemanticIndexStatus:
        return semantic_index_status(
            self.catalog.database_path, self.index_path, self.spec
        )

    def reset_model(self) -> None:
        self._embedder = None

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            if self._embedder_factory is not None:
                self._embedder = self._embedder_factory()
            else:
                self._embedder = FastEmbedder(self.model_cache, self.spec)
        return self._embedder

    @staticmethod
    def _sort_matches(
        matches: Sequence[SearchMatch], sort_order: SortOrder
    ) -> list[SearchMatch]:
        ordered = list(matches)
        if sort_order == "relevance":
            return ordered
        if sort_order not in ("newest", "oldest"):
            raise ValueError(f"unsupported sort order: {sort_order}")

        def date_key(indexed: tuple[int, SearchMatch]) -> tuple[bool, int, int]:
            original_index, match = indexed
            modified = match.result.source_modified_ns
            value = modified or 0
            if sort_order == "newest":
                value = -value
            return modified is None, value, original_index

        return [item for _, item in sorted(enumerate(ordered), key=date_key)]

    def search(
        self,
        query: str,
        mode: SearchMode,
        group: str | None = None,
        visible_limit: int = SEMANTIC_PAGE_SIZE,
        sort_order: SortOrder = "relevance",
    ) -> SearchPage:
        text = query.strip()
        if visible_limit < 1:
            raise ValueError("visible_limit must be positive")
        if mode == "literal":
            results = self.catalog.search(text, group)
            matches = self._sort_matches(
                [SearchMatch(result=item) for item in results], sort_order
            )
            return SearchPage(
                mode=mode,
                query=text,
                matches=tuple(matches),
                total=len(results),
                visible_limit=len(results),
            )
        if mode != "semantic":
            raise ValueError(f"unsupported search mode: {mode}")
        if not text:
            return SearchPage(mode=mode, query=text, matches=(), total=0, visible_limit=visible_limit)

        status = self.index_status()
        if not status.ready:
            raise SemanticIndexUnavailable(status)
        ranking_limit = (
            status.indexed_count
            if sort_order == "relevance"
            else min(status.indexed_count, SEMANTIC_DATE_CANDIDATE_LIMIT)
        )
        ranked = semantic_search(
            self.index_path,
            text,
            self._get_embedder(),
            self.spec,
            limit=ranking_limit,
        )
        available = {
            result.image_id: result for result in self.catalog.search("", group)
        }
        matches = self._sort_matches([
            SearchMatch(result=available[item.image_id], semantic_score=item.score)
            for item in ranked
            if item.image_id in available
        ], sort_order)
        return SearchPage(
            mode=mode,
            query=text,
            matches=tuple(matches[:visible_limit]),
            total=len(matches),
            visible_limit=visible_limit,
        )

    def find_similar(
        self,
        source_image_id: str,
        group: str | None = None,
        visible_limit: int = SEMANTIC_PAGE_SIZE,
        sort_order: SortOrder = "relevance",
    ) -> SearchPage:
        if visible_limit < 1:
            raise ValueError("visible_limit must be positive")
        status = self.index_status()
        if not status.ready:
            raise SemanticIndexUnavailable(status)
        ranking_limit = (
            status.indexed_count
            if sort_order == "relevance"
            else min(status.indexed_count, SEMANTIC_DATE_CANDIDATE_LIMIT)
        )
        try:
            ranked = semantic_neighbors(
                self.index_path,
                source_image_id,
                self.spec,
                limit=ranking_limit,
            )
        except LookupError as error:
            raise SemanticSourceUnavailable(source_image_id) from error
        available = {
            result.image_id: result for result in self.catalog.search("", group)
        }
        matches = self._sort_matches(
            [
                SearchMatch(result=available[item.image_id], semantic_score=item.score)
                for item in ranked
                if item.image_id in available
            ],
            sort_order,
        )
        return SearchPage(
            mode="similar",
            query=source_image_id,
            matches=tuple(matches[:visible_limit]),
            total=len(matches),
            visible_limit=visible_limit,
        )


def build_application_semantic_index(
    catalog_path: Path,
    project_root: Path,
    rebuild: bool = True,
) -> BuildSummary:
    spec = application_spec()
    embedder = FastEmbedder(project_root / "models" / "fastembed", spec)
    return build_trial_index(
        catalog_path,
        project_root / "data" / SEMANTIC_INDEX_FILENAME,
        embedder,
        spec,
        rebuild=rebuild,
    )
