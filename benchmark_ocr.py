from __future__ import annotations

import argparse
import difflib
import html
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_CACHE = PROJECT_ROOT / "models"
TRIAL_DATA = RESULTS_DIR / "ocr_trial.json"
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(MODEL_CACHE))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


def normalize(text: str) -> str:
    return "".join(character for character in text if CJK_PATTERN.match(character))


def similarity(reference: str, candidate: str) -> float:
    reference_normalized = normalize(reference)
    candidate_normalized = normalize(candidate)
    if not reference_normalized:
        return 1.0 if not candidate_normalized else 0.0
    return difflib.SequenceMatcher(None, reference_normalized, candidate_normalized).ratio()


def payload(result) -> dict:
    value = result.json
    return value.get("res", value)


def extract_subtitle(result, image_height: int, crop_mode: bool) -> str:
    data = payload(result)
    lines = []
    for text, score, box in zip(
        data.get("rec_texts") or [],
        data.get("rec_scores") or [],
        data.get("rec_boxes") or [],
    ):
        text = str(text).strip()
        if not text or float(score) < 0.35 or not CJK_PATTERN.search(text):
            continue
        center_y = (float(box[1]) + float(box[3])) / 2
        if crop_mode or center_y >= image_height * 0.45:
            lines.append(text)
    return "\n".join(dict.fromkeys(lines))


def make_lower_crop(path: Path) -> tuple[np.ndarray, int]:
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
        return cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR), crop.height


def timed_prediction(ocr, input_value, image_height: int, crop_mode: bool) -> tuple[float, str]:
    started = time.perf_counter()
    predictions = list(ocr.predict(input_value))
    elapsed = time.perf_counter() - started
    text = extract_subtitle(predictions[0], image_height, crop_mode) if predictions else ""
    return elapsed, text


def aggregate(rows: list[dict], name: str) -> dict:
    matches = [row for row in rows if row["configuration"] == name]
    return {
        "configuration": name,
        "total_seconds": round(sum(row["seconds"] for row in matches), 3),
        "average_seconds": round(statistics.mean(row["seconds"] for row in matches), 3),
        "median_seconds": round(statistics.median(row["seconds"] for row in matches), 3),
        "average_similarity": round(statistics.mean(row["similarity"] for row in matches), 4),
        "exact_matches": sum(row["similarity"] == 1.0 for row in matches),
    }


def build_report(report: dict) -> str:
    summary_rows = "".join(
        f"<tr><td>{html.escape(row['configuration'])}</td><td>{row['average_seconds']:.2f}s</td>"
        f"<td>{row['median_seconds']:.2f}s</td><td>{row['average_similarity']:.1%}</td>"
        f"<td>{row['exact_matches']}/{report['sample_count']}</td></tr>"
        for row in report["summary"]
    )
    detail_rows = "".join(
        f"<tr><td>{row['sample']}</td><td>{html.escape(row['configuration'])}</td>"
        f"<td>{row['seconds']:.2f}s</td><td>{row['similarity']:.1%}</td>"
        f"<td>{html.escape(row['text']).replace(chr(10), '<br>') or '<em>None</em>'}</td></tr>"
        for row in report["rows"]
    )
    winner = report["recommendation"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Image Finder — OCR Speed Benchmark</title>
<style>
:root{{--bg:#f4f1ea;--card:#fffdf8;--ink:#28251f;--muted:#70695e;--line:#ddd4c6;--accent:#a13f2c}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,"Segoe UI",sans-serif}}
main{{width:min(1120px,calc(100% - 30px));margin:42px auto 70px}} h1{{font:700 clamp(2rem,5vw,4rem)/1 Georgia,serif;margin:0 0 8px}}
h2{{font:700 1.35rem Georgia,serif}} .card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;margin:18px 0;box-shadow:0 7px 24px #392b1910}}
.recommend{{border:2px solid var(--accent)}} table{{width:100%;border-collapse:collapse}} th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{color:var(--muted);font-size:.8rem;text-transform:uppercase}} code{{background:#eee6da;padding:2px 5px;border-radius:4px}} .scroll{{overflow-x:auto}}
@media(prefers-color-scheme:dark){{:root{{--bg:#191816;--card:#23211e;--ink:#f2ede4;--muted:#bcb2a4;--line:#403b34;--accent:#f09178}}code{{background:#39342d}}}}
</style></head><body><main>
<h1>OCR Speed Benchmark</h1>
<p>User-selected OCR trial samples. Model initialization and downloads are excluded from inference times. Text similarity is measured against the saved server-model result.</p>
<section class="card recommend"><h2>Recommendation</h2><p>{html.escape(winner)}</p></section>
<section class="card"><h2>Summary</h2><div class="scroll"><table><thead><tr><th>Configuration</th><th>Average</th><th>Median</th><th>Text similarity</th><th>Exact</th></tr></thead><tbody>{summary_rows}</tbody></table></div></section>
<section class="card"><h2>Per-image results</h2><div class="scroll"><table><thead><tr><th>Sample</th><th>Configuration</th><th>Time</th><th>Similarity</th><th>Recognized subtitle</th></tr></thead><tbody>{detail_rows}</tbody></table></div></section>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-data", type=Path, default=TRIAL_DATA)
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        required=True,
        help="one-based sample numbers from the local OCR trial",
    )
    args = parser.parse_args()

    from paddleocr import PaddleOCR

    trial = json.loads(args.trial_data.read_text(encoding="utf-8"))
    if any(number < 1 or number > len(trial["items"]) for number in args.samples):
        raise SystemExit("every --samples value must identify an item in the trial")
    samples = [(number, trial["items"][number - 1]) for number in args.samples]
    rows: list[dict] = []

    for number, item in samples:
        rows.append(
            {
                "sample": number,
                "configuration": "Server / full frame (baseline)",
                "seconds": float(item["elapsed_seconds"]),
                "text": item["subtitle_text"],
                "reference": item["subtitle_text"],
                "similarity": 1.0,
            }
        )

    common = dict(
        lang="ch",
        ocr_version="PP-OCRv5",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
        device="cpu",
    )

    print("Loading server model...", flush=True)
    server = PaddleOCR(**common)
    for number, item in samples:
        path = Path(item["absolute_path"])
        crop, crop_height = make_lower_crop(path)
        seconds, text = timed_prediction(server, crop, crop_height, True)
        rows.append(
            {
                "sample": number,
                "configuration": "Server / lower 55%",
                "seconds": round(seconds, 3),
                "text": text,
                "reference": item["subtitle_text"],
                "similarity": round(similarity(item["subtitle_text"], text), 4),
            }
        )
        print(f"Server crop sample {number}: {seconds:.2f}s", flush=True)

    del server
    print("Loading mobile model...", flush=True)
    mobile = PaddleOCR(
        **common,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
    )
    for number, item in samples:
        path = Path(item["absolute_path"])
        seconds, text = timed_prediction(mobile, str(path), int(item["height"]), False)
        rows.append(
            {
                "sample": number,
                "configuration": "Mobile / full frame",
                "seconds": round(seconds, 3),
                "text": text,
                "reference": item["subtitle_text"],
                "similarity": round(similarity(item["subtitle_text"], text), 4),
            }
        )
        print(f"Mobile full sample {number}: {seconds:.2f}s", flush=True)

    for number, item in samples:
        path = Path(item["absolute_path"])
        crop, crop_height = make_lower_crop(path)
        seconds, text = timed_prediction(mobile, crop, crop_height, True)
        rows.append(
            {
                "sample": number,
                "configuration": "Mobile / lower 55%",
                "seconds": round(seconds, 3),
                "text": text,
                "reference": item["subtitle_text"],
                "similarity": round(similarity(item["subtitle_text"], text), 4),
            }
        )
        print(f"Mobile crop sample {number}: {seconds:.2f}s", flush=True)

    names = [
        "Server / full frame (baseline)",
        "Server / lower 55%",
        "Mobile / full frame",
        "Mobile / lower 55%",
    ]
    summary = [aggregate(rows, name) for name in names]
    eligible = [row for row in summary if row["average_similarity"] >= 0.95]
    winner = min(eligible or summary, key=lambda row: row["average_seconds"])
    baseline = summary[0]
    speedup = baseline["average_seconds"] / winner["average_seconds"]
    recommendation = (
        f"Use {winner['configuration']} as the first-pass candidate. It retained "
        f"{winner['average_similarity']:.1%} average text similarity and was {speedup:.2f}× "
        "the speed of the current baseline on this controlled sample. Keep the server model as a fallback for low-confidence or empty results."
    )
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "samples": args.samples,
        "sample_count": len(samples),
        "method": "User-selected OCR trial samples; inference only; OCR text compared with the saved baseline output.",
        "summary": summary,
        "rows": rows,
        "recommendation": recommendation,
    }
    (RESULTS_DIR / "speed_benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (PROJECT_ROOT / "speed-benchmark.html").write_text(build_report(report), encoding="utf-8")
    print(json.dumps({"summary": summary, "recommendation": recommendation}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
