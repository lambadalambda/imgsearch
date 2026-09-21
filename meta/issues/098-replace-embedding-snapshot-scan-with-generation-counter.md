# Replace the embedding snapshot `MAX(updated_at)` scan with a generation counter

## Summary

`embeddingSnapshot` in `internal/vectorindex/sqlitevector/index.go:88,201-213` runs `MAX(updated_at)` over `image_embeddings` on every search to decide whether to requantize. There is no index on `updated_at`, and the column sits after the 4KB `vector_blob`, so SQLite walks every overflow page. EXPLAIN shows a full `SCAN image_embeddings`. At 100k images that is hundreds of MB of page reads per query.

## Requirements

- Track a generation counter in the index (bumped by `Upsert` and `Delete`, which already invalidate `quantized` in-process) and use it for the requantize decision.
- For cross-process safety (split `-mode=api` / `-mode=worker`), fall back to `PRAGMA data_version` or an indexed column rather than a table scan.

## Acceptance Criteria

- Search no longer issues a full-table query on `image_embeddings` when nothing has changed (verify with EXPLAIN QUERY PLAN or a query-count test).
- Requantization still triggers after an upsert or delete, including one made by another process.

## Notes

- Found during the 2026-09-21 review.
