"""Command-line entry point for read-only image discovery."""

from __future__ import annotations

import argparse
from pathlib import Path

from image_finder.catalog import Catalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan an image library read-only")
    parser.add_argument("library", type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    catalog = Catalog(project_root / "data" / "image-finder.sqlite3", project_root)
    try:
        summary = catalog.scan_library(
            args.library,
            lambda update: print(
                f"\rDiscovered {update.discovered:,}: {update.current_path[:90]:90}",
                end="",
                flush=True,
            ),
        )
        print()
        print(summary)
        return 0 if summary.errors == 0 else 2
    finally:
        catalog.close()


if __name__ == "__main__":
    raise SystemExit(main())
