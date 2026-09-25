"""Create a private, ignored sample of collection-derived OCR keywords."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from image_finder.keyword_exploration import keyword_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 200:
        parser.error("--limit must be between 1 and 200")

    project = Path(__file__).resolve().parent
    output = project / "results" / "keyword-exploration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = keyword_report(project / "data" / "image-finder.sqlite3", args.limit)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote private keyword sample: {output}")
    for source, payload in report["sources"].items():
        print(
            f"{source}: {payload['document_count']:,} OCR records, "
            f"{len(payload['candidates'])} candidates"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
