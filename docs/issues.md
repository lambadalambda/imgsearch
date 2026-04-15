# Issues

This is a lightweight working checklist for system issues and cleanup found during the whole-repo review before deeper performance work.

## Open

- [x] P0 Stop read-path annotation requeue loops and state mutation in `GET` handlers: `docs/issues/001-read-path-requeue-loop.md`
- [x] P1 Unify embedding persistence and vector index ownership: `docs/issues/002-embedding-write-ownership.md`
- [ ] P1 Decide and enforce the sqlite-vector multi-model strategy: `docs/issues/003-sqlite-vector-model-strategy.md`
- [ ] P1 Add explicit runtime modes for embed-only and annotation-capable processes: `docs/issues/004-runtime-modes-and-process-shape.md`
- [x] P1 Enable SQLite foreign key enforcement: `docs/issues/005-enable-sqlite-foreign-keys.md`
- [x] P2 Harden native runtime concurrency and cancellation behavior: `docs/issues/006-native-runtime-safety.md`
- [x] P2 Clean up the search hot path before performance tuning: `docs/issues/007-search-hot-path-cleanup.md`
- [ ] P2 Extract shared helpers and repeated rules: `docs/issues/008-shared-helper-dedup.md`
- [x] P2 Harden default model downloads and startup network behavior: `docs/issues/009-default-model-download-hardening.md`
- [x] P3 Refresh architecture and decision docs after the native-only cutover: `docs/issues/010-doc-drift-after-native-only.md`

## Notes

- The most important pre-performance fixes are the read-path requeue loop, the split ownership of embedding persistence, and the sqlite-vector multi-model story.
- `docs/indexing-annotation-pipeline-notes.md` already captures the longer-term direction for splitting `embed_image` and `annotate_image`; several issues below are prerequisites for that work.
