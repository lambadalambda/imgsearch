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
- [ ] P2 Batched embedding inference via multi-sequence llama.cpp: `docs/issues/014-batched-embedding-inference.md` (partial groundwork landed: `EmbedImages` API + worker wiring, but no stable native throughput win yet)
- [ ] P3 Pipeline CPU preprocessing with GPU inference: `docs/issues/015-pipeline-cpu-gpu-overlap.md`

## Features

- [x] P2 Add MVP video search via sampled representative frames: `docs/issues/016-video-search-via-sampled-frames.md`
- [ ] P2 Add transcript-backed video search with integrated local ASR: `docs/issues/017-video-transcript-asr-spike.md`

## Notes

- The most important pre-performance fixes are the read-path requeue loop, the split ownership of embedding persistence, and the sqlite-vector multi-model story.
- `docs/indexing-annotation-pipeline-notes.md` already captures the longer-term direction for splitting `embed_image` and `annotate_image`; several issues below are prerequisites for that work.
- Performance issues 011-015 are ordered by priority and dependency: 011 should be done first (unblocks VRAM for batching), then 012-013 (prerequisites for 014-015), then 014 (highest potential gain), and finally 015 (low expected return).
- Annotation batching was considered and deprioritized: variable-length autoregressive generation + JSON grammar constraints make it extremely complex for ~10-25% marginal gain.
