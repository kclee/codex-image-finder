(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.SemanticReview = api;
    if (root.document) {
      root.addEventListener("DOMContentLoaded", function () {
        api.initialize(root);
      });
    }
  }
})(typeof window !== "undefined" ? window : null, function () {
  "use strict";

  const REVIEW_FORMAT_VERSION = "image-finder-semantic-human-review-v1";
  const ALLOWED_RATINGS = new Set(["useful", "maybe", "not_useful"]);

  function storageKey(context) {
    return [
      "image-finder.semantic-review",
      REVIEW_FORMAT_VERSION,
      String(context.evaluation_report_version),
      context.query_set_version,
      context.sample_manifest_sha256,
    ].join(".");
  }

  function itemMap(context) {
    return new Map(context.items.map(function (item) {
      return [item.review_key, item];
    }));
  }

  function sameValue(left, right) {
    return JSON.stringify(left) === JSON.stringify(right);
  }

  function validateImport(document, context) {
    const errors = [];
    if (!document || typeof document !== "object") {
      return { ok: false, errors: ["The selected file is not a review document."], ratings: {} };
    }
    if (document.review_format_version !== REVIEW_FORMAT_VERSION) {
      errors.push("Review format version does not match this page.");
    }
    if (document.evaluation_report_version !== context.evaluation_report_version) {
      errors.push("Evaluation report version does not match this page.");
    }
    if (document.sample_manifest_sha256 !== context.sample_manifest_sha256) {
      errors.push("Sample manifest does not match this page.");
    }
    if (document.query_set_version !== context.query_set_version) {
      errors.push("Query-set version does not match this page.");
    }
    if (!Array.isArray(document.ratings)) {
      errors.push("The review document does not contain a ratings array.");
    }
    if (errors.length) {
      return { ok: false, errors: errors, ratings: {} };
    }

    const expected = itemMap(context);
    const ratings = {};
    const identityFields = [
      "query",
      "model_key",
      "model_identifier",
      "model_revision",
      "image_id",
      "image_content_sha256",
      "relative_path",
      "ocr_subtitle",
      "rank",
      "similarity_score",
    ];
    for (const imported of document.ratings) {
      if (!imported || typeof imported !== "object") {
        errors.push("A rating entry is not an object.");
        continue;
      }
      const expectedItem = expected.get(imported.review_key);
      if (!expectedItem) {
        errors.push("A rating refers to a result that is not present in this report.");
        continue;
      }
      if (!ALLOWED_RATINGS.has(imported.human_rating)) {
        errors.push("A rating has an unsupported human_rating value.");
        continue;
      }
      if (Object.prototype.hasOwnProperty.call(ratings, imported.review_key)) {
        errors.push("The review document contains a duplicate result identity.");
        continue;
      }
      const mismatch = identityFields.find(function (field) {
        return !sameValue(imported[field], expectedItem[field]);
      });
      if (mismatch) {
        errors.push("A rating has mismatched result context: " + mismatch + ".");
        continue;
      }
      ratings[imported.review_key] = imported.human_rating;
    }
    return { ok: errors.length === 0, errors: errors, ratings: ratings };
  }

  function buildExport(context, ratings, exportedAt) {
    const rows = [];
    for (const item of context.items) {
      const rating = ratings[item.review_key];
      if (!ALLOWED_RATINGS.has(rating)) {
        continue;
      }
      rows.push(Object.assign({}, item, { human_rating: rating }));
    }
    return {
      review_format_version: REVIEW_FORMAT_VERSION,
      evaluation_report_version: context.evaluation_report_version,
      report_generated_at: context.report_generated_at,
      query_set_version: context.query_set_version,
      sample_manifest_sha256: context.sample_manifest_sha256,
      total_result_count: context.total_result_count,
      reviewed_result_count: rows.length,
      exported_at: exportedAt,
      ratings: rows,
    };
  }

  function saveRatings(storage, context, ratings, savedAt) {
    const payload = buildExport(context, ratings, savedAt);
    storage.setItem(storageKey(context), JSON.stringify(payload));
    return payload;
  }

  function loadRatings(storage, context) {
    const raw = storage.getItem(storageKey(context));
    if (!raw) {
      return { ok: true, errors: [], ratings: {} };
    }
    try {
      return validateImport(JSON.parse(raw), context);
    } catch (error) {
      return { ok: false, errors: ["Saved browser review data is not valid JSON."], ratings: {} };
    }
  }

  function suggestedFileName(context, reviewedCount, date) {
    const status = reviewedCount === context.total_result_count ? "complete" : "partial";
    const reviewed = String(reviewedCount).padStart(3, "0");
    const total = String(context.total_result_count).padStart(3, "0");
    const shortManifest = context.sample_manifest_sha256.slice(0, 8);
    return [
      "image-finder",
      "semantic-model-shootout",
      context.query_set_version.replace(/[^a-z0-9_-]+/gi, "-"),
      shortManifest,
      status + "-" + reviewed + "-of-" + total,
      date,
    ].join("-") + ".json";
  }

  function summarize(ratings, total) {
    const counts = { useful: 0, maybe: 0, not_useful: 0 };
    for (const value of Object.values(ratings)) {
      if (Object.prototype.hasOwnProperty.call(counts, value)) {
        counts[value] += 1;
      }
    }
    const reviewed = counts.useful + counts.maybe + counts.not_useful;
    return Object.assign(counts, { reviewed: reviewed, remaining: total - reviewed });
  }

  function initialize(windowObject) {
    const documentObject = windowObject.document;
    const contextNode = documentObject.getElementById("review-context");
    if (!contextNode) {
      return;
    }
    const context = JSON.parse(contextNode.textContent);
    let ratings = {};
    let storageAvailable = true;
    const status = documentObject.getElementById("review-status");

    function showStatus(message, isError) {
      status.textContent = message;
      status.classList.toggle("error", Boolean(isError));
    }

    try {
      const restored = loadRatings(windowObject.localStorage, context);
      if (restored.ok) {
        ratings = restored.ratings;
      } else {
        storageAvailable = false;
        showStatus("Saved review data was not restored: " + restored.errors.join(" "), true);
      }
    } catch (error) {
      storageAvailable = false;
      showStatus("Browser local storage is unavailable. Export often to preserve your work.", true);
    }

    function updateProgress() {
      const counts = summarize(ratings, context.total_result_count);
      documentObject.getElementById("reviewed-count").textContent = String(counts.reviewed);
      documentObject.getElementById("remaining-count").textContent = String(counts.remaining);
      documentObject.getElementById("useful-count").textContent = String(counts.useful);
      documentObject.getElementById("maybe-count").textContent = String(counts.maybe);
      documentObject.getElementById("not-useful-count").textContent = String(counts.not_useful);
      documentObject.getElementById("review-progress").value = counts.reviewed;
      documentObject.getElementById("export-review").textContent =
        "Export review results (" + counts.reviewed + ")";
      return counts;
    }

    function persist(message) {
      if (!storageAvailable) {
        updateProgress();
        return;
      }
      try {
        saveRatings(windowObject.localStorage, context, ratings, new Date().toISOString());
        showStatus(message || "Saved automatically in this browser.", false);
      } catch (error) {
        storageAvailable = false;
        showStatus("Automatic saving failed. Export your review to preserve it.", true);
      }
      updateProgress();
    }

    for (const input of documentObject.querySelectorAll('input[type="radio"][data-review-key]')) {
      if (ratings[input.dataset.reviewKey] === input.value) {
        input.checked = true;
      }
      input.addEventListener("change", function () {
        if (input.checked) {
          ratings[input.dataset.reviewKey] = input.value;
          persist();
        }
      });
    }

    documentObject.getElementById("export-review").addEventListener("click", function () {
      const counts = updateProgress();
      const exportedAt = new Date().toISOString();
      const payload = buildExport(context, ratings, exportedAt);
      const blob = new Blob([JSON.stringify(payload, null, 2) + "\n"], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = documentObject.createElement("a");
      anchor.href = url;
      anchor.download = suggestedFileName(context, counts.reviewed, exportedAt.slice(0, 10));
      documentObject.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      showStatus("Exported " + counts.reviewed + " reviewed results.", false);
    });

    const importInput = documentObject.getElementById("import-review-file");
    documentObject.getElementById("import-review").addEventListener("click", function () {
      importInput.click();
    });
    importInput.addEventListener("change", async function () {
      const file = importInput.files && importInput.files[0];
      importInput.value = "";
      if (!file) {
        return;
      }
      try {
        const imported = JSON.parse(await file.text());
        const validation = validateImport(imported, context);
        if (!validation.ok) {
          showStatus("Import rejected: " + validation.errors.join(" "), true);
          return;
        }
        ratings = Object.assign({}, ratings, validation.ratings);
        for (const input of documentObject.querySelectorAll('input[type="radio"][data-review-key]')) {
          input.checked = ratings[input.dataset.reviewKey] === input.value;
        }
        persist("Imported and restored " + Object.keys(validation.ratings).length + " ratings.");
      } catch (error) {
        showStatus("Import rejected: the selected file is not valid review JSON.", true);
      }
    });

    documentObject.getElementById("reset-review").addEventListener("click", function () {
      if (!windowObject.confirm("Clear all review ratings saved for this report? This cannot be undone unless you exported them.")) {
        return;
      }
      ratings = {};
      for (const input of documentObject.querySelectorAll('input[type="radio"][data-review-key]')) {
        input.checked = false;
      }
      try {
        windowObject.localStorage.removeItem(storageKey(context));
      } catch (error) {
        storageAvailable = false;
      }
      updateProgress();
      showStatus("Review progress cleared.", false);
    });

    const restoredCount = updateProgress().reviewed;
    if (!status.textContent) {
      showStatus(
        restoredCount ? "Restored " + restoredCount + " saved ratings." : "Ready. Ratings save automatically in this browser.",
        false
      );
    }
  }

  return {
    REVIEW_FORMAT_VERSION: REVIEW_FORMAT_VERSION,
    buildExport: buildExport,
    loadRatings: loadRatings,
    saveRatings: saveRatings,
    storageKey: storageKey,
    suggestedFileName: suggestedFileName,
    summarize: summarize,
    validateImport: validateImport,
    initialize: initialize,
  };
});
