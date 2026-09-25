"""UI-independent domain records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SearchResult:
    image_id: str
    relative_path: str
    absolute_path: Path
    thumbnail_path: Path
    group_name: str
    width: int
    height: int
    source_modified_ns: int | None
    all_text: str
    subtitle_text: str
    confidence: float | None
    analysis_run_id: str | None
    analysis_engine: str | None
    analysis_model: str | None
    analysis_pipeline: str | None
    analysis_created_at: str | None

    @property
    def display_text(self) -> str:
        return self.subtitle_text or self.all_text or "(No recognized text)"


@dataclass(frozen=True, slots=True)
class AnalysisRecord:
    run_id: str
    analysis_type: str
    engine_name: str
    engine_version: str
    model_name: str
    model_version: str
    pipeline_version: str
    created_at: str
    all_text: str
    subtitle_text: str
    confidence: float | None

    @property
    def display_text(self) -> str:
        return self.subtitle_text or self.all_text or "(No recognized text)"


@dataclass(frozen=True, slots=True)
class AnalysisTiming:
    sample_count: int
    median_seconds: float
    mean_seconds: float


@dataclass(frozen=True, slots=True)
class AnalysisBatchSummary:
    queued_at: str
    total: int
    succeeded: int
    pending: int
    running: int
    failed: int
    skipped: int
    inference_seconds: float
