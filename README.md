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
- [`docs/journal/development-history.md`](docs/journal/development-history.md) records
  generalized implementation history, methods, verification, and aggregate findings.
- Private working journals that quote local OCR results remain ignored on the developer's
  machine.
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

## Run Image Finder v0.1

Double-click `start-image-finder.cmd`, or run:

```powershell
.\.venv\Scripts\python.exe run_image_finder.py
```

Image Finder opens the existing local catalog without importing trial data. The main
workflow is intentionally simple:

1. **Text / Literal** searches exact and partial OCR text as you type, including local
   Traditional/Simplified Chinese query variants.
2. **Meaning / Semantic** searches OCR subtitle meaning with the local multilingual
   MiniLM model. Enter a query or choose a preset, then press Enter or **Search**.
3. Meaning search initially shows 24 stable ranked results. **More** adds the next 24
   without changing results already shown.
4. Select a thumbnail for the larger preview. Double-click it or use **Open image** to
   open the original in the default viewer; **Open containing folder** reveals it in
   Explorer for sharing or drag-and-drop.
5. Advanced OCR and review controls remain available under the collapsed **OCR tools**
   section.

The editable meaning presets live in `semantic-presets.json`. The semantic index is
derived local data at `data\semantic-subtitle-v0.1.sqlite3`. If it is missing or stale,
Meaning mode explains why and offers an explicit **Build meaning index** action with an
estimate; ordinary startup, browsing, and literal search never trigger that work. The
same operation is available from the command line:

```powershell
.\.venv\Scripts\python.exe build_semantic_index.py --rebuild
```

The index uses only current non-empty OCR subtitles. It never opens image pixels or
writes to the source collection, and it can be deleted and rebuilt at any time. Model
files remain local under `models\fastembed`; generated data and caches remain ignored
by Git. Delete `data\image-finder.sqlite3` only when intentionally rebuilding the main
catalog.

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
240 MiB. The fixed evaluation queries had no literal matches, making their semantic
results a genuine additional search path.

Results were mixed. Some reaction, paraphrase, and cross-language queries retrieved
conceptually related subtitles, while concepts absent from the deterministic sample
returned unrelated text. Conversational role and participant direction were also weak.

The experiment is promising for paraphrases, reactions, conversational intent, and
cross-language queries when a relevant subtitle exists. It is not reliable for visual
mood, expressions, people, objects, or concepts absent from the subtitle/sample. The
small random sample and OCR noise also create misleading high scores, so no score
threshold has been selected. Exact/partial search remains unchanged and is the primary
route for known wording.

### First usable semantic desktop integration

After the bounded evaluations and human review, v0.1 exposes MiniLM as an explicit
optional **Meaning / Semantic** mode while preserving **Text / Literal** as the default.
It embeds all 2,309 current non-empty OCR subtitles, not image pixels. The application
spec pins the model revision and 384 dimensions described above, mean pooling, L2
normalization, cosine ranking, dynamic-int8 ONNX, a 512-token maximum, a controlled
batch size of 32, the index schema/pipeline version, and the complete source-text
manifest. A change to those settings or the current OCR subtitle text marks the index
stale and requires an explicit rebuild.

The verified local rebuild completed in 466.9 seconds and produced a 4,968,448-byte
SQLite file. A cold semantic search, including local model initialization, took about
1.31 seconds; representative warm searches took about 166–183 ms. Literal search was
independently rechecked, and expanding from 24 to 48 semantic results preserved the
first 24 identities exactly. These measurements are machine-specific. Visual/SigLIP
search remains parked and is not part of the v0.1 desktop application.

### 1,000-subtitle evaluation and experimental hybrid search

The second evaluation retains the same model and storage format but uses a separate
`data\semantic-subtitle-evaluation-1000.sqlite3` index. It selects exactly 1,000 latest
non-empty OCR subtitles by top-level folder: every group receives at least one slot,
remaining slots are allocated proportionally, and content SHA-256 provides the stable
order inside each group. Collection-specific distribution details remain local.

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

The larger sample was more useful than the 50-item sample: reaction, work-avoidance,
and English-to-Chinese conversational queries produced some strong paraphrase matches.
Failures remain important: a comfort query retrieved descriptions of distress rather
than comforting language, keyword-heavy prompts overmatched individual nouns, and an
emotion query returned vague associations. High cosine scores can therefore still be
confidently wrong.

Only `很開心` had literal matches in the fixed query set: seven in the full catalog and
four in the sampled index. Hybrid fusion moved literal matches upward, while the other
15 queries reduced to their semantic ranking. This does not yet establish a generally
better blended ranker. The product implication is to preserve exact/partial search and,
if exposed later, present semantic results as an explicit secondary mode or section.

At the observed throughput, a full local subtitle index was projected to take several
minutes and only a few MiB of vector/index storage, plus the existing model cache. The
full archive was not embedded.

### Controlled embedding-model shootout

The fixed 1,000-subtitle sample and fixed 16-query set were reused unchanged to compare
MiniLM with two deployment-realistic local ONNX alternatives. Run the comparison with:

```powershell
.\.venv\Scripts\python.exe run_semantic_model_shootout.py --rebuild
```

The pinned candidates are `intfloat/multilingual-e5-small` revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3` (384 dimensions, full float32 ONNX,
MIT) and `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` from the
Xenova ONNX repository revision `e5d116277351513fd260955ece953ecddde7046e`
(768 dimensions, dynamic-int8 ONNX, Apache-2.0). E5 uses mean pooling, L2
normalization, and mandatory `query: ` / `passage: ` prefixes. MPNet uses mean pooling
and L2 normalization without prefixes. All model files are local and revision-pinned.

| Model | Package | Embed 1,000 | Throughput | Warm query median / p95 | Raw vectors | SQLite |
|---|---:|---:|---:|---:|---:|---:|
| MiniLM baseline | 240.46 MiB | 113.54 s | 8.81/s | 66.6 / 80.7 ms | 1.46 MiB | 2.07 MiB |
| E5-small | 490.74 MiB | 22.67 s | 44.11/s | 31.1 / 49.8 ms | 1.46 MiB | 2.07 MiB |
| MPNet int8 | 286.87 MiB | 30.23 s | 33.08/s | 55.5 / 69.0 ms | 2.93 MiB | 4.02 MiB |

Default 256-item FastEmbed build batches produced high observed process working-set
peaks: about 2.55 GiB for MiniLM, 6.08 GiB for E5-small, and 7.56 GiB for MPNet. These
are 50 ms observations rather than hard limits, but they are important packaging
evidence: a future desktop build should use smaller controlled embedding batches and
measure memory again before adopting any model. Model load observations were much lower,
roughly 580–810 MiB.

Quality was mixed. MPNet improved one English-to-Chinese conversational query and
removed an opposite-intent result from another query. E5-small was fastest and reduced
one bare-keyword failure, but frequently returned broadly related or noisy text instead
of the requested conversational intent. No model consistently solved participant
direction, and keyword domination remained across candidates.

The complete local top-five rows and timings are generated into the ignored
`results\semantic_model_shootout.json`. Open the ignored `semantic-model-shootout.html`
locally for the side-by-side human review with existing derived thumbnails. Useful/Maybe/Not useful
ratings save automatically to browser `localStorage` for this exact evaluation version,
query set, and sample manifest. The toolbar shows reviewed, remaining, and per-rating
counts. Export creates a descriptive partial or complete JSON filename; importing that
JSON restores progress in another browser or machine after validating the report
version, query set, manifest, model revision, stable image ID, image-content SHA-256,
path, subtitle, rank, and score. Clear review requires confirmation. Browser storage is
profile-local, so export
JSON periodically when the judgments matter.

To change only the static review presentation later without rerunning embeddings or
rankings, use:

```powershell
.\.venv\Scripts\python.exe run_semantic_model_shootout.py --render-existing
```

Cosine values are reference ranking scores only and must not be compared across model
families as probabilities. The completed human review retained MiniLM as the practical
subtitle baseline while confirming that none of the text models is reliable enough to
replace exact/partial search.

### Bounded visual-semantic experiment

The follow-up experiment is also separate from the desktop UI. It uses only the 148
unique image identities already represented in the completed human review and runs the
unchanged 16-query set against local image embeddings:

```powershell
.\.venv\Scripts\python.exe run_visual_semantic_experiment.py --rebuild
```

The selected model is revision-pinned SigLIP 2 Base Patch16 224 using separate int8
ONNX image and text towers. Images are resized directly to 224×224 RGB, normalized with
the recorded SigLIP parameters, embedded to 768-dimensional L2-normalized float32
vectors, and stored in a separate disposable SQLite index. The source images are opened
read-only for inference and then content-hash verified.

The 148-image build took about 31 seconds at 4.84 images/second. The selected local
model package is about 397 MiB, the SQLite index is about 632 KiB, and an unchanged-index
run had about 11 ms median end-to-end query latency. The measured peak process working
set was about 779 MiB; releasing the one-time vision session reduced the post-build
working set substantially. At the same throughput, the current 3,335-image catalog is
estimated at roughly 11.5 minutes and about 9.8 MiB of raw vectors, but it was not run.

Existing labels cover exact reviewed query-image pairs only. Most visual Top-5 results
were new, so they remain explicitly Unreviewed rather than being counted as failures.
The ignored `visual-semantic-reviewed-set.html` report shows visual Top-10 results and
provides the same durable localStorage and JSON export/import workflow. The completed
33-result supplemental review produced 4 Useful, 10 Maybe, and 19 Not useful judgments.
Across the 11 priority queries, visual Top-3 Useful-or-Maybe rates were 40.0% for reaction
queries and 38.9% for conversational intent, below MiniLM's 60.0% and 72.2% respectively.
This does not justify fusion, UI integration, or full-library visual indexing with the
tested model configuration.

## Source control

Downloaded models, the virtual environment, derived data, generated reports and
thumbnails, human review exports, and packaging output are intentionally ignored.
Tracked tests use only synthetic/anonymized fixtures. Before committing, inspect staged
content for OCR text, local paths, hashes, thumbnails, or collection-derived results.
