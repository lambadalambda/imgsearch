# 002 Embedding Write Ownership

## Priority

P1

## Status

Completed.

- `Index.Upsert()` is now the sole owner of `image_embeddings` persistence.
- The worker now calls `Index.Upsert()` before `completeJob()`.
- `completeJob()` now only updates annotations and marks the job done.
- Tests cover the core regression case where index upsert fails and no embedding row should be persisted.

## Summary

Embedding persistence is currently owned by two places. The worker transaction writes `image_embeddings`, and the vector index `Upsert()` call writes `image_embeddings` again. This duplicates work and creates an awkward failure window.

## Why This Matters

- Every completed job performs a redundant embedding write.
- Worker timing logs overstate the separate `db` and `index` phases.
- If index upsert fails after the job transaction commits, the job gets retried even though the embedding row already exists.
- This is a bad shape for the planned `embed_image` / `annotate_image` split.

## Current Behavior

- `internal/worker/queue.go`
- `ProcessOne()` calls `completeJob(...)` and then `q.Index.Upsert(...)`.
- `completeJob(...)` writes `image_embeddings` and marks the job `done`.
- `internal/vectorindex/sqlitevector/index.go`
- `Upsert()` writes `image_embeddings` again.
- `internal/vectorindex/bruteforce/index.go`
- `Upsert()` also writes `image_embeddings`.

## Risks

- Wasted DB work on every job.
- Retrying an already-persisted embedding after vector index failure.
- Unclear transactional semantics.
- Harder migration to separate embedding and annotation job kinds.

## Desired Outcome

- One layer is the single owner of embedding persistence.
- The worker has a clear transaction boundary.
- Failure semantics are easy to understand.
- Instrumentation reflects real work rather than duplicate writes.

## Suggested Approach

- Pick one source of truth for writing `image_embeddings`.
- A likely direction is:
- let `Index.Upsert()` be the sole embedding writer,
- let the worker transaction handle job state and optional annotations only.
- Revisit whether any vector backend-specific initialization needs to remain separate.

## Acceptance Criteria

- Each successful embed job writes the embedding blob once.
- A vector-index failure does not force needless recomputation if the embedding already exists.
- Worker timing logs show meaningful `db` and `index` phases.
- The chosen ownership model is documented in code or docs.

The acceptance criteria are satisfied by the current implementation. A dedicated test for the `Upsert succeeds, completeJob fails` retry path can still be added later as extra coverage, but the current behavior is already safe and idempotent.

## Related Files

- `internal/worker/queue.go`
- `internal/vectorindex/sqlitevector/index.go`
- `internal/vectorindex/bruteforce/index.go`
- `docs/indexing-annotation-pipeline-notes.md`
