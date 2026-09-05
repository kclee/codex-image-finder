# Image Finder

Local, read-only image discovery and search for a personal screenshot archive.

## Built through human-AI collaboration

This project is being designed and developed collaboratively with OpenAI Codex. The
project owner defines the goals, constraints, and product decisions and reviews the
results; Codex assists with exploration, implementation, testing, and documentation.
AI assistance accelerates the work, while important behavior and generated results
remain subject to human review.

The application itself also uses local machine-learning tools for OCR and, in future
versions, semantic image search. This is separate from the use of Codex during
development: the original images remain local and read-only by design.

## Documentation

- [`progress.html`](progress.html) is the concise visual dashboard.
- [`docs/journal/2026-09-05.md`](docs/journal/2026-09-05.md) records the detailed
  implementation history, methods, verification, files, and commit subjects.
- Experiment galleries and benchmarks remain HTML when visual presentation matters.

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
catalog during development, then supports subtitle/text filtering, folder filtering,
preview, and opening the original file. Normal application startup now opens the existing
catalog without importing trial data. Delete `data\image-finder.sqlite3` to rebuild it.

Use **Add / scan folder…** to discover a complete library. Scanning reads source files
but never writes below the selected library root. New or changed files receive an exact
content hash and a derived thumbnail; unchanged files reuse their existing identity and
thumbnail. A moved or renamed file keeps the same image identity while its previous
location remains in the database as history.

For controlled development checks, `run_selected_ocr.py` accepts one or more exact
catalog-relative paths. It queues only those image identities, loads the local
PP-OCRv5 mobile models on demand, and skips results already completed for the same
engine/model/pipeline version. Full-library OCR is not started automatically.

The desktop action **OCR next 10 in view** prepares and processes at most ten
outstanding image identities from the currently visible gallery, in its displayed
path order. Current folder and text filters therefore control the candidates.
Already-completed identities are skipped and existing pending work in the view is
resumed first. **Pause OCR** stops after the current image; unfinished jobs remain in
SQLite for the next run. **Retry failed** explicitly returns failed jobs to the queue.
Opening the application never starts OCR or prepares the complete library.

After a batch, the gallery automatically switches to **Last OCR batch** so the processed
images remain visible. A checkmark identifies results from the current OCR pipeline,
and the inspector reports its engine, model, pipeline version, completion time, and
confidence. Toggle **Last OCR batch** off to return to the full gallery.

When an image has results from multiple analysis versions, the inspector provides a
version selector. Choosing an entry changes the displayed recognized text and metadata
without deleting or overwriting the other versions.

Search accepts either Traditional or Simplified Chinese. The app uses OpenCC locally
at query time to try equivalent script forms while preserving the original OCR output
unchanged. This normalization is part of search, not OCR, so it can be replaced or
enhanced without reprocessing any images.

The intended mature startup behavior separates inexpensive discovery from expensive
analysis: a read-only incremental library scan may run automatically in the background,
while OCR remains visible, resumable, and user-controlled. A future queue preview will
show which images are next, which is useful both during initial import and after adding
new files.

## Source control

Downloaded models, the virtual environment, derived data, generated thumbnails, and
packaging output are intentionally ignored. The repository should normally be private
because reports may contain local paths, filenames, or recognized subtitle text.
