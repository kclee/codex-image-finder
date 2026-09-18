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
  initial implementation history, methods, verification, files, and commit subjects.
- [`docs/journal/2026-09-18.md`](docs/journal/2026-09-18.md) records the bounded
  semantic subtitle-search experiment and measured results.
- Experiment galleries and benchmarks remain HTML when visual presentation matters.

## Data rule

> Source images are permanent. Everything else is derived, replaceable, and rebuildable.

The application never modifies the selected image library. Its machine-generated
catalog, thumbnails, OCR output, indexes, and future embeddings live in the project
data directory and can be rebuilt. Manual review choices are the separate user-authored
exception described below; they remain portable and should be backed up.

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

Both the lower-frame pass and full-frame fallback use the image already decoded by
Pillow rather than asking Paddle/OpenCV to reopen the filename. This also supports
misleading extensions such as GIF content stored under a `.jpg` name without modifying
the source file.

The desktop batch-size selector offers 10, 25, 50, or 100 images and defaults to the
conservative value 10. The **OCR next N in view** action prepares and processes at most
that many outstanding image identities from the currently visible gallery, in its
displayed path order. Current folder and text filters therefore control the candidates.
Already-completed identities are skipped and existing pending work in the view is
resumed first. **Pause OCR** stops after the current image; unfinished jobs remain in
SQLite for the next run. **Retry failed** explicitly returns failed jobs to the queue.
Opening the application never starts OCR or prepares the complete library.

After the OCR approach is accepted, **OCR all remaining (N)** processes every
outstanding stable image identity in the current gallery with one model startup. It is
still explicit and filter-scoped: opening the app never starts the long run, and a
folder or search filter limits what “all” means. The progress bar shows completed and
target counts, while the timing line shows elapsed time and a live ETA. **Pause OCR**
remains safe for this mode. If the window is closed while OCR is active, the app waits
for the current image to be stored and then exits; starting the remaining action later
resumes the persisted pending work instead of duplicating completed results.

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

The inspector provides the two requested sharing actions. **Copy subtitle text** copies
the selected analysis version's CJK subtitle to the system clipboard after converting
it to standard Traditional Chinese with OpenCC; the stored OCR remains unchanged.
The action is disabled when no selected subtitle exists. **Open containing folder**
opens Windows File Explorer with the original image selected, ready to drag into a
browser or another application. The equivalent macOS path uses Finder reveal, while
other platforms open the parent folder.

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

**Needs review** filters the current search/folder scope to completed results from the
current OCR pipeline that have no selected Chinese/Japanese subtitle, no confidence
score, or confidence below 75 percent. The inspector states the review reason. This is
not a failure count: English images and images without subtitles are intentionally
included so the user can classify them visually. In the right inspector, **Accept OCR**,
**Needs correction**, and **Not relevant** save a decision and advance to the next
unreviewed candidate. **Clear decision** removes that choice. The review-status dropdown
can retrieve images by any saved decision. Because these images are already processed,
the next-OCR strip collapses and OCR is disabled until the review filter is turned off.

Unlike rebuildable OCR results, manual review choices are user-authored data. They live
separately in `user-data\review-state.sqlite3`, keyed by stable image identity and OCR
run version, and are intentionally excluded from Git. Back up the `user-data` folder if
those choices matter; close Image Finder before copying it so SQLite has finished all
writes. Reprocessing with a new OCR version creates a new review context without
overwriting decisions made against an older result.

Startup deliberately separates inexpensive discovery from expensive analysis: the
read-only incremental library check runs automatically in the background, while OCR
remains visible, resumable, and user-controlled. The queue preview is useful both during
initial import and after adding new files.

## Semantic subtitle-search trial

The first semantic experiment is deliberately separate from the desktop UI and from
exact/partial text search. It embeds only the latest OCR `subtitle_text` for a
deterministic sample of at most 50 present image identities. It never opens image pixels
or writes to the source library.

The local stack is FastEmbed 0.8.0 with ONNX Runtime 1.29.0 and the quantized ONNX
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model (384 dimensions),
from `qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q` revision
`faf4aa4225822f3bc6376869cb1164e8e3feedd0`. The pipeline uses mean pooling, a 512-token
maximum, L2-normalized float32 vectors, and cosine similarity without query/document
prefixes.

Run the isolated command-line trial after the approved model has been downloaded into
`models\fastembed`:

```powershell
.\.venv\Scripts\python.exe run_semantic_trial.py --rebuild --top-k 5 `
  "想吃甜食" "傻眼" "很尷尬，不知道該說什麼" `
  "朋友吵架後互相道歉" "awkward reaction" "I don't know what to say"
```

The derived index is `data\semantic-subtitle-trial.sqlite3`. Its language-neutral
SQLite rows contain stable image IDs, relative paths, OCR subtitle text, per-text
SHA-256 hashes, little-endian float32 vectors, and JSON version metadata. The run ID
changes when the engine, runtime, model revision, dimensions, pipeline, schema, or
preprocessing parameters change. A manifest covering selected image IDs, paths, and
source-text hashes detects OCR or selection changes. Matching data is reused; `--rebuild`
forces regeneration. The database and model cache are ignored by Git and are safe to
delete and rebuild.

The verified sample contained exactly 50 subtitles. Cached model load took about 1.1
seconds (1.97 seconds on the first process), embedding and index creation took 2.67
seconds, and 30 warm semantic searches had a 38 ms median and 43 ms p95. The SQLite
index is 128 KiB, including 75 KiB of raw vector values; the local model cache is about
240 MiB. All six unmodified evaluation queries had zero exact/partial matches in the
full OCR catalog, making their semantic results a genuine additional search path.

Results were mixed. `傻眼` found `(你是傻瓜吗？)` first (0.771), and both
`很尷尬，不知道該說什麼` and `awkward reaction` found `感觉怪怪的 / 何か変。`
first (0.630 and 0.621). English `I don't know what to say` found several speech-related
Chinese/Japanese subtitles. In contrast, `想吃甜食` returned unrelated results because
the deterministic sample contained no suitable food subtitle, and the friend-apology
scenario produced weak generic associations. A disclosed post-hoc probe,
`安慰心情不好的人`, ranked `別愁眉苦脸的` first (0.634).

The experiment is promising for paraphrases, reactions, conversational intent, and
cross-language queries when a relevant subtitle exists. It is not reliable for visual
mood, expressions, people, objects, or concepts absent from the subtitle/sample. The
small random sample and OCR noise also create misleading high scores, so no score
threshold or desktop integration has been selected yet. Exact/partial search remains
unchanged and should remain the primary route for known wording.

### 1,000-subtitle evaluation and experimental hybrid search

The second evaluation retains the same model and storage format but uses a separate
`data\semantic-subtitle-evaluation-1000.sqlite3` index. It selects exactly 1,000 of the
2,309 latest non-empty OCR subtitles by top-level folder: every group receives at least
one slot, remaining slots are allocated proportionally, and content SHA-256 provides
the stable order inside each group. This includes archive-root images, every Japan year
folder from 2018 through 2023, CN, Anime, Game, Korean, US, stickers, and smaller groups.

The fixed evaluation queries live in `semantic-evaluation-queries.json`. Run the
reproducible evaluation with:

```powershell
.\.venv\Scripts\python.exe run_semantic_evaluation.py --rebuild --limit 1000 --top-k 5
```

The experimental hybrid uses equal-weight reciprocal-rank fusion with `k=60` over the
semantic ranking and exact/partial/OpenCC ranking from the same sampled index. It does
not interpret cosine similarity as probability and does not replace the existing
full-catalog lexical search. Complete top-five results and measurements are recorded in
`results\semantic_evaluation_1000.json`.

Building 1,000 embeddings took 153.25 seconds (6.53 subtitles/second). Cached model
startup took 1.07 seconds. The 2.07 MiB SQLite index contains 1.46 MiB of raw vectors;
the unchanged model cache is 240 MiB. Sixteen warm semantic queries had a 57 ms median,
and lexical fusion increased the median to 63 ms. A matching index was detected in
0.027 seconds, excluding model startup.

The larger sample was more useful than the 50-item sample: strong retrieval included
`被嚇到` → `可怕可怕可怕`, `不想上班` → `我干不下去了`, and English
`I don't know what to say` → `不知道该怎么回复他才好`. Failures remain important:
the comfort query retrieved descriptions of distress rather than comforting language,
`吐槽朋友` mostly matched the concept of “friend,” and `很無奈` returned vague
associations. High cosine scores can therefore still be confidently wrong.

Only `很開心` had literal matches in the fixed query set: seven in the full catalog and
four in the sampled index. Hybrid fusion moved literal matches upward, while the other
15 queries reduced to their semantic ranking. This does not yet establish a generally
better blended ranker. The product implication is to preserve exact/partial search and,
if exposed later, present semantic results as an explicit secondary mode or section.

At the observed throughput, all 2,309 current non-empty subtitles would take roughly
354 seconds (5.9 minutes), about 3.38 MiB of raw vectors, and approximately 4.8 MiB of
SQLite storage, plus the existing model cache. The full archive was not embedded.

## Source control

Downloaded models, the virtual environment, derived data, generated thumbnails, and
packaging output are intentionally ignored. The repository should normally be private
because reports may contain local paths, filenames, or recognized subtitle text.
