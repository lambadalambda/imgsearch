# Issues

This is a lightweight working checklist for system issues and cleanup found during the whole-repo review before deeper performance work.

## Open

- [x] P0 Stop read-path annotation requeue loops and state mutation in `GET` handlers: `docs/issues/001-read-path-requeue-loop.md`
- [x] P1 Unify embedding persistence and vector index ownership: `docs/issues/002-embedding-write-ownership.md`
- [x] P1 Decide and enforce the sqlite-vector multi-model strategy: `docs/issues/003-sqlite-vector-model-strategy.md`
- [x] P1 Add explicit runtime modes for embed-only and annotation-capable processes: `docs/issues/004-runtime-modes-and-process-shape.md`
- [x] P1 Enable SQLite foreign key enforcement: `docs/issues/005-enable-sqlite-foreign-keys.md`
- [x] P2 Harden native runtime concurrency and cancellation behavior: `docs/issues/006-native-runtime-safety.md`
- [x] P2 Clean up the search hot path before performance tuning: `docs/issues/007-search-hot-path-cleanup.md`
- [x] P2 Extract shared helpers and repeated rules: `docs/issues/008-shared-helper-dedup.md`
- [x] P2 Harden default model downloads and startup network behavior: `docs/issues/009-default-model-download-hardening.md`
- [x] P3 Refresh architecture and decision docs after the native-only cutover: `docs/issues/010-doc-drift-after-native-only.md`

## Performance Optimization

- [x] P0 Enable Flash Attention and tune llama.cpp context parameters: `docs/issues/011-flash-attention-and-context-tuning.md`
- [x] P1 Eliminate double image preprocessing round-trip: `docs/issues/012-eliminate-double-preprocessing.md` (cancelled: benchmarked 36% slower on M2 Max, raw pixels 16x larger than JPEG)
- [x] P2 Batch job claiming at the queue level: `docs/issues/013-batch-job-claiming.md`
- [ ] P2 Batched embedding inference via multi-sequence llama.cpp: `docs/issues/014-batched-embedding-inference.md` (partial groundwork landed: `EmbedImages` API + worker wiring + runtime `n_seq_max` wiring, but no stable native throughput win yet)
- [x] P3 Pipeline CPU preprocessing with GPU inference: `docs/issues/015-pipeline-cpu-gpu-overlap.md`

## Features

- [x] P2 Add MVP video search via sampled representative frames: `docs/issues/016-video-search-via-sampled-frames.md`
- [x] P2 Add transcript-backed video search with integrated local ASR: `docs/issues/017-video-transcript-asr-spike.md`
- [x] P2 Tighten UI radius language and chrome hierarchy: `docs/issues/018-ui-radius-and-chrome-tightening.md`
- [x] P2 Standardize library card height and content density: `docs/issues/019-card-height-and-density.md`
- [x] P2 Add hover/focus overlay expansion for clipped card content: `docs/issues/020-card-overlay-expansion.md`
- [x] P3 Move card actions into thumbnail overlays and simplify rest state: `docs/issues/021-card-action-overlay.md`
- [x] P2 Shift to a search-first masthead and quiet ops controls: `docs/issues/022-search-first-masthead-and-quiet-ops.md`
- [x] P2 Add tag cloud and explicit tag search flow: `docs/issues/023-tag-cloud-and-tag-search.md`

## Security & Maintainability

- [x] P0 Require API authentication and route guarding for `/api/*`: `docs/issues/024-api-authentication-and-route-guarding.md`
- [x] P0 Restrict `/media/` to explicit media subdirectories only: `docs/issues/025-restrict-media-file-serving-scope.md`
- [x] P1 Harden multipart upload resource limits and cleanup: `docs/issues/026-harden-multipart-upload-resource-usage.md`
- [x] P1 Add explicit HTTP server timeouts and header limits: `docs/issues/027-add-http-server-timeouts-and-limits.md`
- [x] P1 Bound transcription audio memory and feature allocation: `docs/issues/028-bound-transcription-audio-memory.md`
- [x] P1 Default container bind address to loopback and require opt-in exposure: `docs/issues/029-safe-network-bind-defaults-for-container.md`
- [x] P2 Deduplicate HTTP query parsing helpers: `docs/issues/030-deduplicate-http-query-parsing-helpers.md`
- [x] P2 Deduplicate tag JSON decoding helpers: `docs/issues/031-deduplicate-tags-json-decoding.md`
- [x] P2 Unify item ID parsing and safe stored-path removal helpers: `docs/issues/032-unify-item-id-and-storage-path-helpers.md`
- [x] P2 Merge duplicate worker job processing paths: `docs/issues/033-merge-duplicate-worker-job-processing-paths.md`
- [x] P3 Extract shared `mark job done` SQL helper: `docs/issues/034-extract-shared-mark-job-done-sql.md`
- [x] P2 Centralize NSFW SQL fragments used by list/search queries: `docs/issues/035-centralize-nsfw-sql-fragments.md`

## Whole-App Review Follow-ups

- [x] P1 Harden batch job leasing and worker option defaults: `docs/issues/036-harden-batch-job-leasing.md`
- [x] P2 Extract typed job handlers from worker switch statement: `docs/issues/037-split-worker-job-handlers.md`
- [x] P2 Make `annotate_image` jobs claimable in batch workers: `docs/issues/053-annotate-image-batch-claiming.md`
- [x] P2 Include videos in live updates and stale-request guards: `docs/issues/038-live-updates-include-videos.md`
- [x] P1 Make retry and stats media-aware for video jobs: `docs/issues/039-media-aware-retry-and-stats.md`
- [ ] P2 Narrow native runtime locks around preprocessing: `docs/issues/040-narrow-native-runtime-locks.md`
- [x] P1 Add graceful shutdown and cancellation-aware worker loops: `docs/issues/041-graceful-shutdown-and-worker-cancellation.md`
- [x] P2 Enforce vector dimensions and share cosine logic: `docs/issues/042-vector-dimension-validation-and-cosine.md`
- [x] P2 Tighten API path, method, and dependency semantics: `docs/issues/043-tighten-api-route-semantics.md`
- [x] P2 Clarify multipart batch upload semantics: `docs/issues/044-clarify-multipart-batch-upload-semantics.md`
- [x] P2 Extract shared image/video media operations: `docs/issues/045-extract-shared-media-http-operations.md`
- [x] P2 Wire browser-level UI smoke tests: `docs/issues/046-ui-smoke-tests.md`
- [x] P2 Extract shared frontend collection and card helpers: `docs/issues/051-frontend-collection-and-card-helpers.md`
- [x] P2 Harden scripts/tooling portability and wire script tests: `docs/issues/047-script-tooling-portability.md`
- [x] P2 Extract runtime config from CLI flag parsing: `docs/issues/048-runtime-config-extraction.md`
- [x] P2 Promote `internal/app` into the runtime composition layer: `docs/issues/052-app-runtime-composition-layer.md`
- [ ] P3 Extract repeated native task commands from `mise.toml`: `docs/issues/049-extract-native-task-command-scripts.md`
- [ ] P2 Harden sqlite-vector quantized search under mixed models: `docs/issues/050-sqlite-vector-quantized-mixed-models.md`

## Notes

- The most important pre-performance fixes are the read-path requeue loop, the split ownership of embedding persistence, and the sqlite-vector multi-model story.
- `docs/indexing-annotation-pipeline-notes.md` already captures the longer-term direction for splitting `embed_image` and `annotate_image`; several issues below are prerequisites for that work.
- Performance issues 011-015 are ordered by priority and dependency: 011 should be done first (unblocks VRAM for batching), then 012-013 (prerequisites for 014-015), then 014 (highest potential gain), and finally 015 (low expected return).
- Annotation batching was considered and deprioritized: variable-length autoregressive generation + JSON grammar constraints make it extremely complex for ~10-25% marginal gain.
