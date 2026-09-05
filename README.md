# Image Finder

Local, read-only image discovery and search for a personal screenshot archive.

## Data rule

> Source images are permanent. Everything else is derived, replaceable, and rebuildable.

The application never modifies the selected image library. Its database, thumbnails,
OCR output, indexes, and future embeddings live in the project data directory.

## Architecture rules

- A path is a location, not an image identity. Exact SHA-256 content hashes identify
  unchanged files across moves and renames; perceptual similarity may be added later.
- Every analysis result belongs to a versioned run recording its engine, model,
  pipeline version, parameters, and timestamps.
- Scanning, persistence, analysis, and search do not import PySide6. The desktop UI
  consumes ordinary domain records through an application service.
- Persistent interchange uses SQLite, JSON, and ordinary files rather than Python-only
  serialization such as pickle.

## Run the prototype

Double-click `start-image-finder.cmd`, or run:

```powershell
.\.venv\Scripts\python.exe run_image_finder.py
```

The current prototype imports the existing 40-image OCR trial into a disposable SQLite
catalog, then supports subtitle/text filtering, folder filtering, preview, and opening
the original file. Delete `data\image-finder.sqlite3` to rebuild it.

## Source control

Downloaded models, the virtual environment, derived data, generated thumbnails, and
packaging output are intentionally ignored. The repository should normally be private
because reports may contain local paths, filenames, or recognized subtitle text.
