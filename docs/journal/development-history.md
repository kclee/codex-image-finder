# Image Finder development history

This public journal intentionally excludes source-library paths, filenames, OCR text,
image identities and hashes, thumbnails, generated rankings, and human review data.
Detailed working notes containing those values remain local and ignored.

## Local catalog and OCR workflow

The project established a read-only image-discovery architecture backed by a local
SQLite catalog. Source images are never modified or accompanied by sidecar files.
Thumbnails, OCR payloads, analysis jobs, and review state are stored separately under
the project and can be rebuilt or backed up according to whether they are generated or
user-authored.

Exact and partial OCR text search supports Traditional/Simplified Chinese query
variants. OCR analysis is explicit, resumable, versioned, and independent from folder
discovery. Tests use generated images and synthetic subtitle strings.

## Small semantic subtitle trial

The first semantic experiment embedded only OCR subtitle text for a deterministic
sample capped at 50 images. It did not inspect image pixels and did not replace exact or
partial search. The index records the embedding engine/runtime versions, pinned model
revision, dimensions, pipeline and preprocessing parameters, deterministic sample
manifest, and source-text hashes so stale data is rebuilt safely.

The local 384-dimensional multilingual MiniLM trial showed useful paraphrase,
reaction, and cross-language retrieval when relevant subtitle evidence existed. It
also showed predictable limitations: missing concepts cannot be recovered, OCR noise
can attract rankings, conversational roles may be reversed, and high cosine scores can
still be wrong. Exact/partial search therefore remains the primary mode for known text.

Aggregate performance from the bounded trial was interactive after model load: cached
startup was about one second, a 50-item index built in a few seconds, and warm queries
were measured in tens of milliseconds. Derived SQLite indexes and model caches remain
ignored and disposable.

## Larger evaluation and model shootout

A controlled 1,000-subtitle evaluation reused the same deterministic selection,
versioning, and local-only storage approach. Aggregate measurements showed that index
storage remained small and query latency remained interactive. A lexical/semantic
reciprocal-rank-fusion experiment improved literal matches but did not establish a
universally better blended ranker.

The fixed sample and query set were then used to compare three revision-pinned local
ONNX multilingual embedding candidates: 384-dimensional MiniLM, 384-dimensional
multilingual E5-small, and 768-dimensional multilingual MPNet. E5-small had the highest
measured embedding throughput, MPNet improved some cross-language conversational
queries, and no model eliminated keyword domination or participant-direction errors.
Memory observations indicate that future desktop packaging should use controlled batch
sizes and repeat peak-memory measurements.

Actual rankings, paths, subtitles, thumbnails, hashes, and model-by-image result rows
are generated into ignored local files. They are not repository source.

## Durable static review

The static shootout report supports Useful, Maybe, and Not useful judgments with
automatic browser localStorage persistence. It displays progress counts, exports a
portable JSON backup, validates report/sample/result identity on import, and requires
confirmation before clearing a review. Human judgments are private local data and are
never committed.

The report generator and JavaScript are reusable repository code. Automated tests build
synthetic reports and verify stable identity, persistence, export/import round trips,
partial reviews, incompatibility handling, and reset behavior without reading the local
collection.

## Generalized human-review conclusion

Completed human review confirms that subtitle semantics are useful as an optional
discovery tool, but not reliable enough to replace exact/partial text search. The compact
MiniLM baseline and the larger MPNet candidate were broadly comparable in practical
usefulness; MPNet sometimes widened result coverage, while MiniLM generally provided the
better balance of early precision, acceptance, storage, and observed memory. The faster
E5 candidate produced more broadly plausible than decisively useful results, especially
for conversational intent.

Useful images were not consistently ordered by similarity score, so a future interface
should expose several results and may benefit from an explicit, bounded “show another
batch” discovery action. The next experiment should remain local and small: test whether
visual-semantic reranking improves the already reviewed candidates before integrating
semantic search into the desktop UI or indexing the full collection.

## Bounded visual-semantic follow-up

The visual follow-up used a revision-pinned multilingual SigLIP 2 Base Patch16 224 int8
ONNX package. It embedded only the 148 unique image identities represented in the
completed review, retained the unchanged 16-query set, and wrote a separate versioned
SQLite index containing 768-dimensional float32 vectors. Original files remained
read-only and were content-hash verified after inference.

The fresh bounded build took about 31 seconds at 4.84 images per second. The selected
model package was about 397 MiB, the generated index about 632 KiB, and the observed
process working-set peak about 779 MiB. Releasing the one-time vision graph after the
build materially reduced resident memory. A later unchanged-index process measured
about 11 ms median end-to-end query latency.

The experiment occasionally recovered a human-reviewed Useful result that MiniLM had
missed, but known-label overlap was sparse: most visual Top-5 results had never been
judged for that exact query. They remain Unreviewed rather than being treated as
negative. A private static report therefore requested only 33 supplemental judgments
across the priority reaction and conversational-intent queries. No fusion, application
integration, or full-catalog visual processing was performed.

The completed supplemental review contained 4 Useful, 10 Maybe, and 19 Not useful
ratings. Once combined with the existing exact-pair labels, the visual Top 3 was fully
reviewed for all 11 priority queries. Useful-or-Maybe rates were 40.0% for visual
reaction results and 38.9% for visual conversational-intent results, compared with
60.0% and 72.2% for MiniLM Top 3 on the same query groups. Visual search occasionally
found a useful result, but repeated generic candidates and poor intent discrimination
outweighed that complementary signal. The current decision is to keep the experiment
rebuildable but not proceed to fusion, UI integration, or full-library visual indexing.

## First usable desktop semantic search

The first usable v0.1 application keeps literal OCR search as the default and adds a
clearly separate semantic subtitle mode. The interface exposes editable query presets,
an explicit search action, a deterministic first page of 24 results, and cumulative
24-result expansion. Existing folder filtering, thumbnail selection, preview, copy,
Explorer reveal, OCR, and review behavior remain available. Double-click and a dedicated
button can open the selected original in the operating system's default image viewer.
Advanced OCR controls are collapsed by default to keep the primary search workflow
focused.

Semantic index creation is never automatic. Missing, stale, incompatible, and ready
states are reported explicitly, and the user must confirm the local build. The
language-neutral SQLite index uses only current non-empty OCR subtitles, remains under
the project data directory, and records the pinned engine/runtime/model revision,
dimensions, pooling and normalization, batch size, pipeline/schema version, and a
manifest of stable identities and source-text hashes. Original image bytes are neither
opened for embedding nor changed. Tests use only generated images and synthetic text.

The verified application index contains 2,309 subtitles and was rebuilt with a
controlled batch size of 32 in 466.9 seconds. Its SQLite file is 4,968,448 bytes. Cold
semantic search, including cached local model initialization, measured about 1.31
seconds; representative warm searches measured about 166–183 ms. The first 24 results
remained an exact prefix after expansion to 48. Literal search and OpenCC behavior were
rechecked independently. The bounded visual experiment remains parked: v0.1 does not
load SigLIP, embed pixels, fuse rankings, or add server/cloud components.
