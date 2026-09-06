"""Offline Chinese-script query expansion while preserving raw OCR text."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _converters():  # type: ignore[no-untyped-def]
    import opencc

    return tuple(
        opencc.OpenCC(configuration)
        for configuration in ("t2s.json", "s2t.json", "tw2s.json", "s2tw.json")
    )


@lru_cache(maxsize=512)
def query_variants(query: str) -> tuple[str, ...]:
    text = query.strip()
    if not text:
        return ()
    candidates = [text]
    candidates.extend(converter.convert(text) for converter in _converters())
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


@lru_cache(maxsize=512)
def to_traditional(text: str) -> str:
    """Convert copied OCR text to standard Traditional Chinese."""

    return _converters()[1].convert(text)
