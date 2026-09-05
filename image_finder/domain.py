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
    all_text: str
    subtitle_text: str
    confidence: float | None

    @property
    def display_text(self) -> str:
        return self.subtitle_text or self.all_text or "(No recognized text)"
