"""UI-independent rules for highlighting OCR results that merit human review."""

from __future__ import annotations


LOW_CONFIDENCE_THRESHOLD = 0.75


def ocr_review_reason(
    subtitle_text: str, confidence: float | None
) -> str | None:
    if not subtitle_text.strip():
        return "No Chinese/Japanese subtitle was selected"
    if confidence is None:
        return "No confidence score is available"
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return f"Low OCR confidence ({confidence:.1%})"
    return None
