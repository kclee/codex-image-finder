"""Application composition root."""

from __future__ import annotations

import argparse
from pathlib import Path

from .catalog import Catalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Local screenshot search prototype")
    parser.add_argument("--smoke-test", action="store_true", help="Open briefly and exit")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    catalog = Catalog(project_root / "data" / "image-finder.sqlite3", project_root)
    try:
        from .ui import run

        return run(catalog, smoke_test=args.smoke_test)
    finally:
        catalog.close()


if __name__ == "__main__":
    raise SystemExit(main())
