# 007 Search Hot Path Cleanup

## Priority

P2

## Status

Completed.

- Negative-prompt embedding now runs sequentially instead of using fake concurrency around a mutex-serialized embedder.
- Search result enrichment now batch-loads image metadata with one `IN (...)` query.
- Result order, duplicate hits, and hit distances are preserved after the batch load.
- Tests cover the no-concurrency expectation, duplicate/order preservation, and the zero-hit enrich case.

## Summary

The search handler has a couple of non-essential inefficiencies that are worth simplifying before deeper profiling work: unnecessary concurrent negative/query embedding and N+1 image metadata loading.

## Why This Matters

- Search is directly user-facing.
- Small hot-path inefficiencies complicate benchmarking and tracing.
- The current code suggests concurrency where the native runtime likely serializes access anyway.

## Current Behavior

- `internal/search/http.go`
- `embedQueryVector()` starts two goroutines when a negative prompt is present.
- The native embedder path is mutex-protected, so this is likely serialized internally.
- `enrich()` loads image metadata row-by-row instead of batching.

## Desired Outcome

- Simpler query embedding flow.
- Batched metadata enrichment.
- Cleaner search benchmarks.

## Suggested Approach

- Replace the dual goroutine negative/query embedding with sequential calls unless real parallelism becomes possible.
- Batch-load image metadata for search hits with one `IN (...)` query.

## Acceptance Criteria

- Search code is simpler and easier to profile.
- Hit enrichment no longer performs one query per result row.
- Tests still cover both text search and similar-image search behavior.

All acceptance criteria are satisfied by the current implementation.

## Related Files

- `internal/search/http.go`
- `internal/search/http_test.go`
