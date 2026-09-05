"""Versioned PaddleOCR subtitle adapter with no UI dependencies."""

from __future__ import annotations

import os
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .analysis_queue import AnalysisSpec


CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
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


@dataclass(frozen=True, slots=True)
class OcrOutput:
    all_text: str
    subtitle_text: str
    confidence: float | None
    payload: dict[str, object]


def _result_payload(result) -> dict:  # Paddle result objects are intentionally isolated here
    value = result.json
    return value.get("res", value)


def _extract_lines(payload: dict, *, lower_only: bool, image_height: int) -> list[dict[str, object]]:
    lines: list[dict[str, object]] = []
    for text, score, box in zip(
        payload.get("rec_texts") or [],
        payload.get("rec_scores") or [],
        payload.get("rec_boxes") or [],
    ):
        normalized = str(text).strip()
        confidence = float(score)
        coordinates = [int(round(float(value))) for value in box]
        if not normalized or confidence < 0.35:
            continue
        center_y = (coordinates[1] + coordinates[3]) / 2
        if lower_only and center_y < image_height * 0.45:
            continue
        lines.append(
            {
                "text": normalized,
                "confidence": confidence,
                "box": coordinates,
                "contains_cjk": bool(CJK_PATTERN.search(normalized)),
            }
        )
    return lines


class PaddleSubtitleOcr:
    def __init__(self, model_cache: Path) -> None:
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(model_cache))
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            device="cpu",
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
        )

    def analyze(self, path: Path) -> OcrOutput:
        started = time.perf_counter()
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            original_height = image.height
            crop_y = round(image.height * 0.45)
            crop = image.crop((0, crop_y, image.width, image.height))
            scale = min(2.0, max(1.0, 1600 / max(1, crop.width)))
            if scale > 1.0:
                crop = crop.resize(
                    (round(crop.width * scale), round(crop.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            crop_array = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)

        predictions = list(self.ocr.predict(crop_array))
        pass_name = "lower-55-percent"
        lines = _extract_lines(
            _result_payload(predictions[0]) if predictions else {},
            lower_only=False,
            image_height=crop.height,
        )
        subtitle_lines = [line for line in lines if line["contains_cjk"]]

        if not subtitle_lines:
            predictions = list(self.ocr.predict(str(path)))
            pass_name = "full-frame-fallback"
            lines = _extract_lines(
                _result_payload(predictions[0]) if predictions else {},
                lower_only=False,
                image_height=original_height,
            )
            subtitle_lines = [
                line
                for line in lines
                if line["contains_cjk"]
                and (line["box"][1] + line["box"][3]) / 2 >= original_height * 0.45
            ]

        selected_confidences = [
            float(line["confidence"]) for line in (subtitle_lines or lines)
        ]
        elapsed = time.perf_counter() - started
        return OcrOutput(
            all_text="\n".join(dict.fromkeys(str(line["text"]) for line in lines)),
            subtitle_text="\n".join(
                dict.fromkeys(str(line["text"]) for line in subtitle_lines)
            ),
            confidence=(
                statistics.mean(selected_confidences) if selected_confidences else None
            ),
            payload={
                "pipeline_version": MOBILE_SUBTITLE_SPEC.pipeline_version,
                "pass": pass_name,
                "elapsed_seconds": round(elapsed, 4),
                "lines": lines,
            },
        )
