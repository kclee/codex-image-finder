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
