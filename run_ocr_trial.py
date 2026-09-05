from __future__ import annotations

import hashlib
import html
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


ARCHIVE_ROOT = Path(r"C:\Users\kckck\Desktop\Output")
PROJECT_ROOT = Path(r"C:\D_Data\AI-Projects\image-finder")
RESULTS_DIR = PROJECT_ROOT / "results"
THUMBNAIL_DIR = RESULTS_DIR / "thumbnails"
MODEL_CACHE = PROJECT_ROOT / "models"
TRIAL_SIZE = 40
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(MODEL_CACHE))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


def stable_key(path: Path) -> str:
    relative = path.relative_to(ARCHIVE_ROOT).as_posix().casefold()
    return hashlib.sha256(relative.encode("utf-8")).hexdigest()


def top_group(path: Path) -> str:
    relative = path.relative_to(ARCHIVE_ROOT)
    return relative.parts[0] if len(relative.parts) > 1 else "(archive root)"


def collect_images() -> list[dict]:
    records: list[dict] = []
    for path in ARCHIVE_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception:
            continue
        orientation = "landscape" if width > height else "portrait" if height > width else "square"
        records.append(
            {
                "path": path,
                "relative_path": path.relative_to(ARCHIVE_ROOT).as_posix(),
                "group": top_group(path),
                "extension": path.suffix.lower(),
                "width": width,
                "height": height,
                "orientation": orientation,
                "bytes": path.stat().st_size,
            }
        )
    return records


def choose_sample(records: list[dict], limit: int) -> list[dict]:
    """Deterministic round-robin sample across top-level archive groups."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["group"]].append(record)
    for group_records in groups.values():
        group_records.sort(key=lambda item: stable_key(item["path"]))

    ordered_groups = sorted(groups, key=lambda name: (-len(groups[name]), name.casefold()))
    selected: list[dict] = []
    index = 0
    while len(selected) < limit:
        added = False
        for group in ordered_groups:
            if index < len(groups[group]):
                selected.append(groups[group][index])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        index += 1

    # Ensure rare orientations and formats are represented when available.
    selected_paths = {item["path"] for item in selected}
    for field, values in (("orientation", ("portrait", "square")), ("extension", (".gif", ".bmp"))):
        for value in values:
            if any(item[field] == value for item in selected):
                continue
            candidates = sorted(
                (item for item in records if item[field] == value and item["path"] not in selected_paths),
                key=lambda item: stable_key(item["path"]),
            )
            if candidates and selected:
                removed = selected.pop()
                selected_paths.remove(removed["path"])
                selected.append(candidates[0])
                selected_paths.add(candidates[0]["path"])
    return selected


def result_payload(result) -> dict:
    payload = result.json
    return payload.get("res", payload)


def extract_lines(payload: dict, source: str, y_offset: int = 0, scale: float = 1.0) -> list[dict]:
    texts = payload.get("rec_texts") or []
    scores = payload.get("rec_scores") or []
    boxes = payload.get("rec_boxes") or []
    lines: list[dict] = []
    for text, score, box in zip(texts, scores, boxes):
        x1, y1, x2, y2 = [float(value) for value in box]
        normalized_box = [
            round(x1 / scale),
            round(y1 / scale + y_offset),
            round(x2 / scale),
            round(y2 / scale + y_offset),
        ]
        lines.append(
            {
                "text": str(text).strip(),
                "confidence": float(score),
                "box": normalized_box,
                "source": source,
                "contains_cjk": bool(CJK_PATTERN.search(str(text))),
            }
        )
    return [line for line in lines if line["text"]]


def deduplicate_lines(lines: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    order: list[str] = []
    for line in lines:
        key = re.sub(r"\s+", "", line["text"]).casefold()
        if not key:
            continue
        if key not in best:
            best[key] = line
            order.append(key)
        elif line["confidence"] > best[key]["confidence"]:
            best[key] = line
    return [best[key] for key in order]


def make_thumbnail(path: Path, lines: list[dict], output_path: Path) -> None:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        stroke = max(2, round(max(image.size) / 500))
        for line in lines:
            color = "#ff4d3d" if line["source"] == "full" else "#24c98a"
            draw.rectangle(line["box"], outline=color, width=stroke)
        image.thumbnail((720, 480), Image.Resampling.LANCZOS)
        image.save(output_path, "JPEG", quality=86, optimize=True)


def analyze_image(ocr, record: dict, ordinal: int) -> dict:
    path: Path = record["path"]
    started = time.perf_counter()
    lines: list[dict] = []
    error = None
    used_lower_crop = False
    try:
        predictions = list(ocr.predict(str(path)))
        if predictions:
            lines.extend(extract_lines(result_payload(predictions[0]), "full"))

        cjk_in_lower_half = any(
            line["contains_cjk"] and ((line["box"][1] + line["box"][3]) / 2) >= record["height"] * 0.45
            for line in lines
        )
        if not cjk_in_lower_half:
            used_lower_crop = True
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                crop_y = round(image.height * 0.45)
                crop = image.crop((0, crop_y, image.width, image.height))
                scale = min(2.0, max(1.0, 1600 / max(1, crop.width)))
                if scale > 1.0:
                    crop = crop.resize(
                        (round(crop.width * scale), round(crop.height * scale)),
                        Image.Resampling.LANCZOS,
                    )
                crop_array = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)
            crop_predictions = list(ocr.predict(crop_array))
            if crop_predictions:
                lines.extend(extract_lines(result_payload(crop_predictions[0]), "lower-crop", crop_y, scale))
        lines = deduplicate_lines(lines)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - started
    lower_lines = [
        line
        for line in lines
        if ((line["box"][1] + line["box"][3]) / 2) >= record["height"] * 0.45
    ]
    subtitle_lines = [line for line in lower_lines if line["contains_cjk"] and line["confidence"] >= 0.35]
    thumbnail_name = f"sample-{ordinal:02d}.jpg"
    try:
        make_thumbnail(path, lines, THUMBNAIL_DIR / thumbnail_name)
    except Exception as exc:
        error = error or f"ThumbnailError: {exc}"

    return {
        **{key: value for key, value in record.items() if key != "path"},
        "absolute_path": str(path),
        "thumbnail": f"results/thumbnails/{thumbnail_name}",
        "elapsed_seconds": round(elapsed, 3),
        "used_lower_crop": used_lower_crop,
        "all_text": "\n".join(line["text"] for line in lines),
        "subtitle_text": "\n".join(line["text"] for line in subtitle_lines),
        "line_count": len(lines),
        "cjk_line_count": sum(1 for line in lines if line["contains_cjk"]),
        "average_confidence": round(statistics.mean(line["confidence"] for line in lines), 4) if lines else None,
        "lines": lines,
        "error": error,
    }


def build_report(payload: dict) -> str:
    cards = []
    for index, item in enumerate(payload["items"], 1):
        subtitle = html.escape(item["subtitle_text"]) or "<span class='muted'>No confident lower-frame Chinese subtitle found</span>"
        all_text = html.escape(item["all_text"]) or "No text detected"
        path_uri = Path(item["absolute_path"]).as_uri()
        status = "CJK found" if item["cjk_line_count"] else "No CJK detected"
        cards.append(
            f"""
            <article class="card">
              <a href="{html.escape(path_uri)}"><img src="{html.escape(item['thumbnail'])}" alt="OCR sample {index}"></a>
              <div class="body">
                <div class="meta"><span>#{index:02d}</span><span>{html.escape(item['group'])}</span><span>{item['width']}×{item['height']}</span></div>
                <h2>{html.escape(item['relative_path'])}</h2>
                <p class="status">{status} · {item['elapsed_seconds']:.1f}s</p>
                <h3>Likely subtitle</h3>
                <p class="ocr">{subtitle.replace(chr(10), '<br>')}</p>
                <details><summary>All detected text ({item['line_count']} lines)</summary><p class="ocr">{all_text.replace(chr(10), '<br>')}</p></details>
              </div>
            </article>
            """
        )
    summary = payload["summary"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Image Finder — OCR Trial</title>
  <style>
    :root {{ --bg:#f4f1ea; --card:#fffdf8; --ink:#28251f; --muted:#746d62; --line:#ded5c7; --accent:#a13f2c; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,"Segoe UI",sans-serif; }}
    main {{ width:min(1500px,calc(100% - 28px)); margin:34px auto 70px; }}
    header {{ max-width:900px; margin-bottom:24px; }}
    h1 {{ margin:0; font:700 clamp(2rem,5vw,4rem)/1.05 Georgia,serif; }}
    .summary {{ color:var(--muted); font-size:1.05rem; }}
    .legend {{ display:flex; gap:18px; flex-wrap:wrap; margin:14px 0; }}
    .dot {{ display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:6px; }}
    .red {{ background:#ff4d3d; }} .green {{ background:#24c98a; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(310px,1fr)); gap:18px; }}
    .card {{ overflow:hidden; border:1px solid var(--line); border-radius:13px; background:var(--card); box-shadow:0 7px 24px #392b1910; }}
    img {{ display:block; width:100%; aspect-ratio:3/2; object-fit:contain; background:#171717; }}
    .body {{ padding:16px; }}
    .meta {{ display:flex; flex-wrap:wrap; gap:6px; }}
    .meta span {{ padding:3px 7px; border-radius:999px; background:#eee6da; font-size:.78rem; }}
    h2 {{ margin:10px 0; font-size:.92rem; overflow-wrap:anywhere; }}
    h3 {{ margin:13px 0 4px; font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
    .status {{ color:var(--accent); font-weight:700; }}
    .ocr {{ min-height:1.5em; font-size:1.08rem; }}
    .muted {{ color:var(--muted); }}
    details {{ border-top:1px solid var(--line); padding-top:9px; }}
    summary {{ cursor:pointer; color:var(--muted); }}
    @media (prefers-color-scheme:dark) {{
      :root {{ --bg:#191816; --card:#23211e; --ink:#f2ede4; --muted:#bcb2a4; --line:#403b34; --accent:#f09178; }}
      .meta span {{ background:#39342d; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>OCR Trial</h1>
      <p class="summary">{summary['sample_count']} images · {summary['images_with_any_text']} with text · {summary['images_with_cjk']} with Chinese characters · {summary['images_with_likely_subtitles']} with likely lower-frame Chinese subtitles · {summary['errors']} errors · {summary['elapsed_seconds']:.1f}s total</p>
      <p>This is an automated trial, not a final accuracy score. Compare the recognized text with each thumbnail. Red boxes came from full-frame OCR; green boxes came from the lower-frame fallback.</p>
      <div class="legend"><span><i class="dot red"></i>Full frame</span><span><i class="dot green"></i>Lower-frame fallback</span></div>
    </header>
    <section class="grid">{''.join(cards)}</section>
  </main>
</body>
</html>"""


def main() -> None:
    from paddleocr import PaddleOCR

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    records = collect_images()
    sample = choose_sample(records, TRIAL_SIZE)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "archive_root": str(ARCHIVE_ROOT),
        "method": "Deterministic round-robin across top-level folders; rare orientations and formats forced into sample.",
        "sample_size": len(sample),
        "items": [{key: value for key, value in item.items() if key != "path"} | {"absolute_path": str(item["path"])} for item in sample],
    }
    (RESULTS_DIR / "sample_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    ocr = PaddleOCR(
        lang="ch",
        ocr_version="PP-OCRv5",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
        device="cpu",
    )

    partial = RESULTS_DIR / "ocr_trial.partial.json"
    analyzed: list[dict] = []
    if partial.exists():
        try:
            checkpoint = json.loads(partial.read_text(encoding="utf-8"))
            completed = checkpoint.get("items", [])
            expected_paths = [str(item["path"]) for item in sample[: len(completed)]]
            completed_paths = [item.get("absolute_path") for item in completed]
            if completed_paths == expected_paths:
                analyzed = completed
                print(f"Resuming from checkpoint after {len(analyzed)} images.", flush=True)
        except (OSError, ValueError, TypeError):
            analyzed = []

    for ordinal, record in enumerate(sample[len(analyzed) :], len(analyzed) + 1):
        print(f"[{ordinal:02d}/{len(sample)}] {record['relative_path']}", flush=True)
        item = analyze_image(ocr, record, ordinal)
        analyzed.append(item)
        checkpoint = {"items": analyzed}
        partial.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    elapsed = sum(float(item.get("elapsed_seconds", 0)) for item in analyzed)
    summary = {
        "sample_count": len(analyzed),
        "images_with_any_text": sum(bool(item["line_count"]) for item in analyzed),
        "images_with_cjk": sum(bool(item["cjk_line_count"]) for item in analyzed),
        "images_with_likely_subtitles": sum(bool(item["subtitle_text"]) for item in analyzed),
        "lower_crop_fallbacks": sum(bool(item["used_lower_crop"]) for item in analyzed),
        "errors": sum(bool(item["error"]) for item in analyzed),
        "elapsed_seconds": round(elapsed, 3),
        "average_seconds_per_image": round(elapsed / len(analyzed), 3) if analyzed else None,
    }
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "configuration": {
            "engine": "PaddleOCR 3.7.0 / PP-OCRv5 server models",
            "language": "ch",
            "device": "CPU",
            "enable_mkldnn": False,
            "archive_modified": False,
        },
        "summary": summary,
        "items": analyzed,
    }
    (RESULTS_DIR / "ocr_trial.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (PROJECT_ROOT / "ocr-trial.html").write_text(build_report(payload), encoding="utf-8")
    if partial.exists():
        partial.unlink()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
