# 001 Read-Path Requeue Loop

## Priority

P0

## Status

Completed via the short-term stopgap:
- `GET /api/images` and `/api/live` no longer requeue jobs.
- Missing annotation repair now happens through the explicit `POST /api/jobs/retry-failed` action.
- Searchability is no longer revoked by merely browsing the UI.

## Summary

Read paths are currently mutating indexing state when annotations are missing. Simply browsing the gallery or leaving the live view open can requeue completed jobs, reset retry state, and trigger unnecessary re-embedding work.

## Why This Matters

- `GET` handlers should not write to job state.
- Failed annotation attempts become hard to reason about because `attempts` and `last_error` get reset.
- Viewing the UI can create background churn and distort `/api/stats`.
- This makes performance measurements noisy and makes the planned pipeline split harder.

## Current Behavior

- `internal/images/http.go`
- `List()` marks images with `index_state == "done"` and missing annotations as requeue candidates.
- It then calls `imgdb.RequeueDoneJob(...)` before returning the response.
- `internal/live/http.go`
- `writeSnapshot()` calls `images.List()` on every push interval, so keeping the live UI open repeatedly triggers the same requeue logic.
- `internal/db/index_jobs.go`
- `RequeueDoneJob()` and `RequeueDoneJobsMissingAnnotations()` reset `attempts`, clear `run_after`, clear `last_error`, and move jobs back to `pending`.
- `internal/worker/queue.go`
- Annotation errors are logged, but the job still completes and becomes eligible for requeue again.

## Risks

- Endless re-embedding for images that consistently fail annotation.
- Hidden failure history.
- Write contention on the single SQLite connection.
- Search and similar-image UI state becomes misleading because annotation gaps can turn a searchable image back into `pending`.

## Desired Outcome

- No `GET` path mutates `index_jobs`.
- Missing annotations are handled by explicit background repair or a dedicated annotation queue.
- Searchability is not revoked just because annotations are missing.
- Retry state remains meaningful.

## Suggested Approach

- Short-term stopgap:
- Remove requeue behavior from `images.List()`.
- Keep completed embedding jobs as `done` even when annotations are missing.
- Trigger missing-annotation repair from an explicit admin action or startup/background repair pass.
- Better long-term direction:
- Split `embed_image` and `annotate_image`.
- Represent searchable and annotated state separately.

## Acceptance Criteria

- Opening `/api/images` and `/api/live` causes no writes to `index_jobs`.
- Annotation failures do not automatically reset attempts to zero from read paths.
- Search results remain available for images whose annotations are still missing.
- Queue metrics remain stable while the UI is open.

All of the above are satisfied by the current short-term fix. The longer-term `annotate_image` split is still tracked separately in `docs/indexing-annotation-pipeline-notes.md` and the remaining follow-up issues.

## Related Files

- `internal/images/http.go`
- `internal/live/http.go`
- `internal/db/index_jobs.go`
- `internal/worker/queue.go`
- `docs/indexing-annotation-pipeline-notes.md`
