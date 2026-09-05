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

On later normal launches, the app automatically checks the most recently registered
library in a background thread after the window opens. Existing gallery results remain
usable while discovery runs; OCR controls are temporarily disabled to avoid competing
analysis. Unchanged files are recognized by stored size and modification time and are
not rehashed. If no library is registered, or the saved folder is unavailable, the app
stays open with a helpful status message. Diagnostic smoke tests suppress this scan.

For controlled development checks, `run_selected_ocr.py` accepts one or more exact
catalog-relative paths. It queues only those image identities, loads the local
PP-OCRv5 mobile models on demand, and skips results already completed for the same
engine/model/pipeline version. Add `--pause-after N` to stop cleanly after N attempts;
running the same selection again resumes its persisted pending jobs without duplicating
completed work. Console text is escaped safely so a Windows code-page limitation cannot
turn a successfully stored Chinese result into a reported OCR failure. Full-library OCR
is not started automatically.

The desktop batch-size selector offers 10, 25, 50, or 100 images and defaults to the
conservative value 10. The **OCR next N in view** action prepares and processes at most
that many outstanding image identities from the currently visible gallery, in its
displayed path order. Current folder and text filters therefore control the candidates.
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

The **Next to OCR** strip previews up to the first ten thumbnails from the exact upcoming
batch, along with the selected batch count, total eligible count, and approximate number
of batches. It uses the queue's read-only candidate calculation, so merely viewing it
does not create jobs. The strip stays fixed while a batch runs and refreshes after
completion. Beneath it, measured median inference time estimates the selected batch and
remaining current view; model-loading overhead and the estimated number of loads are
called out separately instead of being hidden in the estimate. A larger selected batch
reuses one model initialization but remains pausable after the current image.

During OCR, the progress bar reports both exact count and percentage, for example
`Current batch: 17 / 25 images (68%)`. The estimate uses two plain-language lines:
**Typical OCR** is the median inference time from completed images; **Selected batch**
and **Remaining** multiply that measurement by the relevant image counts. **One model
startup** means initializing the already-downloaded Paddle detection and recognition
models in memory once for that button-triggered batch, not downloading a model or
loading one model per image. The gallery count is labeled separately from OCR progress.

**OCR history…** shows recent persisted queue groups with total, complete, pending,
running, failed, skipped, and measured inference-time columns. These summaries use
existing job timestamps and are operational history rather than a permanent audit log.

Startup deliberately separates inexpensive discovery from expensive analysis: the
read-only incremental library check runs automatically in the background, while OCR
remains visible, resumable, and user-controlled. The queue preview is useful both during
initial import and after adding new files.

## Source control

Downloaded models, the virtual environment, derived data, generated thumbnails, and
packaging output are intentionally ignored. The repository should normally be private
because reports may contain local paths, filenames, or recognized subtitle text.
