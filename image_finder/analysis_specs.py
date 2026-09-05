"""Lightweight declarations of versioned analysis configurations."""

from .analysis_queue import AnalysisSpec


MOBILE_SUBTITLE_SPEC = AnalysisSpec(
    analysis_type="ocr",
    engine_name="PaddleOCR",
    engine_version="3.7.0",
    model_name="PP-OCRv5_mobile_det+PP-OCRv5_mobile_rec",
    model_version="PP-OCRv5",
    pipeline_version="subtitle-lower55-mobile-v1",
    parameters={
        "language": "ch",
        "device": "cpu",
        "crop_start": 0.45,
        "minimum_confidence": 0.35,
        "fallback": "mobile-full-frame-when-no-cjk",
        "mkldnn": False,
    },
)
