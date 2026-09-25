"""Bounded, deterministic keyword exploration over existing OCR text."""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

from .text_search import to_traditional


KeywordSource = Literal["subtitle_text", "all_text"]
_CJK_RUN = re.compile(r"[\u3400-\u9fff]{2,}")
_STOP_PHRASES = {
    "一直",
    "一個",
    "一些",
    "一定",
    "不是",
    "但是",
    "事情",
    "什麼",
    "他們",
    "你的",
    "你們",
    "個人",
    "可以",
    "可能",
    "因為",
    "因爲",
    "如果",
    "好的",
    "好了",
    "就是",
    "已經",
    "怎麼",
    "我的",
    "我也",
    "是不",
    "是我",
    "我們",
    "所以",
    "時候",
    "的人",
    "的事",
    "的時",
    "的話",
    "沒有",
    "然後",
    "現在",
    "真的",
    "真是",
    "自己",
    "覺得",
    "這個",
    "這麼",
    "這種",
    "這樣",
    "那個",
    "還是",
    "不會",
    "不要",
    "下去",
    "只要",
    "大家",
    "知道",
    "起來",
    "都是",
    "東西",
}


@dataclass(frozen=True, slots=True)
class KeywordCandidate:
    text: str
    document_count: int
    occurrence_count: int
    score: float


def _is_candidate(text: str) -> bool:
    if text in _STOP_PHRASES or len(set(text)) == 1:
        return False
    return not any(stop in text for stop in _STOP_PHRASES if len(stop) < len(text))


def extract_keyword_candidates(
    texts: Iterable[str],
    *,
    minimum_documents: int = 4,
    maximum_document_ratio: float = 0.08,
    limit: int = 40,
) -> list[KeywordCandidate]:
    """Return recurring CJK n-grams without retaining source text or identities."""

    normalized = [to_traditional(text) for text in texts if text.strip()]
    occurrences: Counter[str] = Counter()
    documents: Counter[str] = Counter()
    for text in normalized:
        seen: set[str] = set()
        for run in _CJK_RUN.findall(text):
            for width in (2, 3, 4):
                for start in range(0, len(run) - width + 1):
                    term = run[start : start + width]
                    if _is_candidate(term):
                        occurrences[term] += 1
                        seen.add(term)
        documents.update(seen)

    document_total = len(normalized)
    maximum_documents = max(
        minimum_documents, int(document_total * maximum_document_ratio)
    )
    ranked = []
    for term, document_count in documents.items():
        if not minimum_documents <= document_count <= maximum_documents:
            continue
        length_bonus = 1.0 + 0.35 * (len(term) - 2)
        ranked.append(
            KeywordCandidate(
                text=term,
                document_count=document_count,
                occurrence_count=occurrences[term],
                score=round(document_count * length_bonus, 3),
            )
        )
    ranked.sort(
        key=lambda item: (-item.score, -len(item.text), -item.document_count, item.text)
    )

    selected: list[KeywordCandidate] = []
    for candidate in ranked:
        if any(
            candidate.text in existing.text
            or existing.text in candidate.text
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def latest_ocr_texts(catalog_path: Path, source: KeywordSource) -> list[str]:
    if source not in ("subtitle_text", "all_text"):
        raise ValueError(f"unsupported keyword source: {source}")
    uri = catalog_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            f"""
            SELECT ar.{source}
            FROM images i
            JOIN analysis_results ar ON ar.id=(
                SELECT ar2.id
                FROM analysis_results ar2
                JOIN analysis_runs run2 ON run2.id=ar2.run_id
                WHERE ar2.image_id=i.id
                ORDER BY COALESCE(run2.completed_at, run2.started_at) DESC, ar2.id DESC
                LIMIT 1
            )
            WHERE EXISTS (
                SELECT 1 FROM file_locations fl
                WHERE fl.image_id=i.id AND fl.is_present=1
            ) AND trim(ar.{source})<>''
            ORDER BY i.content_sha256
            """
        ).fetchall()
    finally:
        connection.close()
    return [str(row[0]) for row in rows]


def keyword_report(catalog_path: Path, limit: int = 40) -> dict[str, object]:
    sources: dict[str, object] = {}
    for source in ("subtitle_text", "all_text"):
        texts = latest_ocr_texts(catalog_path, source)
        sources[source] = {
            "document_count": len(texts),
            "candidates": [
                asdict(candidate)
                for candidate in extract_keyword_candidates(texts, limit=limit)
            ],
        }
    return {
        "report_version": "bounded-cjk-ngram-v1",
        "normalization": "OpenCC Traditional Chinese",
        "ngram_widths": [2, 3, 4],
        "minimum_documents": 4,
        "maximum_document_ratio": 0.08,
        "contains_source_paths": False,
        "contains_source_text": False,
        "sources": sources,
    }
