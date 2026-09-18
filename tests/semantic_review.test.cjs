"use strict";

const assert = require("node:assert/strict");
const review = require("../semantic_review.js");

class MemoryStorage {
  constructor() {
    this.values = new Map();
  }
  getItem(key) {
    return this.values.has(key) ? this.values.get(key) : null;
  }
  setItem(key, value) {
    this.values.set(key, String(value));
  }
  removeItem(key) {
    this.values.delete(key);
  }
}

const context = {
  evaluation_report_version: 1,
  report_generated_at: "2026-09-18T11:43:11-0500",
  query_set_version: "reaction-intents-v1",
  sample_manifest_sha256: "838f8dfbee9be2d2b414e5e1775e1683938500ec827de3b645148c2109c126f8",
  total_result_count: 2,
  items: [
    {
      review_key: "q0|minilm|revision-a|image-a",
      query: "合成查詢",
      model_key: "minilm",
      model_identifier: "sentence-transformers/example",
      model_revision: "revision-a",
      image_id: "image-a",
      image_content_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      relative_path: "a.png",
      ocr_subtitle: "合成字幕甲",
      rank: 1,
      similarity_score: 0.8,
    },
    {
      review_key: "q0|mpnet|revision-b|image-b",
      query: "合成查詢",
      model_key: "mpnet",
      model_identifier: "sentence-transformers/example-2",
      model_revision: "revision-b",
      image_id: "image-b",
      image_content_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      relative_path: "b.png",
      ocr_subtitle: "合成字幕乙",
      rank: 1,
      similarity_score: 0.7,
    },
  ],
};

const storage = new MemoryStorage();
const partial = { "q0|minilm|revision-a|image-a": "useful" };
review.saveRatings(storage, context, partial, "2026-09-18T12:00:00Z");
assert.deepEqual(review.loadRatings(storage, context).ratings, partial);

const exported = review.buildExport(context, partial, "2026-09-18T12:01:00Z");
assert.equal(exported.reviewed_result_count, 1);
assert.equal(exported.total_result_count, 2);
assert.equal(exported.ratings[0].image_id, "image-a");
assert.equal(exported.ratings[0].image_content_sha256.length, 64);
assert.equal(exported.ratings[0].human_rating, "useful");
const imported = review.validateImport(JSON.parse(JSON.stringify(exported)), context);
assert.equal(imported.ok, true);
assert.deepEqual(imported.ratings, partial);

const counts = review.summarize(partial, 2);
assert.deepEqual(counts, { useful: 1, maybe: 0, not_useful: 0, reviewed: 1, remaining: 1 });
assert.match(
  review.suggestedFileName(context, 1, "2026-09-18"),
  /reaction-intents-v1-838f8dfb-partial-001-of-002-2026-09-18\.json$/
);
assert.match(
  review.suggestedFileName(context, 2, "2026-09-18"),
  /complete-002-of-002-2026-09-18\.json$/
);

const wrongManifest = JSON.parse(JSON.stringify(exported));
wrongManifest.sample_manifest_sha256 = "different";
assert.equal(review.validateImport(wrongManifest, context).ok, false);

const wrongIdentity = JSON.parse(JSON.stringify(exported));
wrongIdentity.ratings[0].image_id = "different-image";
assert.equal(review.validateImport(wrongIdentity, context).ok, false);

storage.removeItem(review.storageKey(context));
assert.deepEqual(review.loadRatings(storage, context).ratings, {});

console.log("semantic review JavaScript checks passed");
