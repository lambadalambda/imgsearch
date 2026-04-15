# 003 sqlite-vector Model Strategy

## Priority

P1

## Summary

The sqlite-vector backend currently scans all stored embeddings and filters by `model_id` afterward. That creates unnecessary work and makes mixed model histories risky, especially if dimensions differ between model versions.

## Why This Matters

- Old embedding rows remain in `image_embeddings` as model versions change.
- Search work grows with total historical embeddings, not just the active model.
- Mixed dimensions in a single vector column/table may produce invalid or fragile behavior.
- This must be decided before serious performance tuning.

## Current Behavior

- `internal/vectorindex/sqlitevector/index.go`
- `Search()` uses `vector_full_scan('image_embeddings', 'vector_blob', ?, ?)` and filters with `WHERE ie.model_id = ?` afterward.
- `countEmbeddings()` counts all rows in `image_embeddings`, not rows for the active model.
- `ensureInitialized()` initializes the shared `image_embeddings.vector_blob` table/index for a given dimension.

## Open Design Question

What do we want to support?

- One active embedding model at a time, with old rows purged.
- Multiple retained model versions, but isolated safely per model.
- Multiple retained model versions only when dimensions stay identical.

## Desired Outcome

- Explicit, documented policy for old embedding rows.
- sqlite-vector queries only scan the rows they should.
- Safe behavior when model versions or dimensions change.

## Suggested Approach

- Decide one of these paths before optimization work:
- purge old embeddings when activating a new model version,
- or isolate embeddings per model in separate tables or another safe layout,
- or explicitly forbid mixed-dimension retention and enforce that rule.

## Acceptance Criteria

- The chosen model/version policy is documented.
- Search cost scales with the active model's rows, not all historical rows.
- Tests cover at least one multi-model scenario.
- There is no ambiguous mixed-dimension behavior left in sqlite-vector.

## Related Files

- `internal/vectorindex/sqlitevector/index.go`
- `internal/db/models.go`
- `internal/db/index_jobs.go`
- `docs/indexing-annotation-pipeline-notes.md`
