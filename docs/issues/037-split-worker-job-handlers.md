# 037: Extract Typed Job Handlers From Worker Switch Statement

## Priority

P2

## Status

Open.

## Summary

The worker queue still owns claiming, job dispatch, media-specific execution, retry policy, completion SQL, and logging in one large file. Extract typed job handlers and define shared job kind constants to reduce drift.

## Context

- `internal/worker/queue.go` contains the queue type, claim logic, execution switch, media loaders, retry handling, and completion helpers.
- Job kind strings such as `embed_image`, `annotate_image`, `annotate_video`, and `transcribe_video` are repeated across worker, DB, and HTTP packages.
- `executeClaimedJob` in `internal/worker/queue.go:460` has a large switch over job kind with media-specific side effects.

## Risks

- String typos can create unreachable job paths or orphaned queue rows.
- Worker behavior changes are difficult to review because unrelated concerns are colocated.
- Future job kinds will expand an already large switch and increase test setup cost.

## Acceptance Criteria

- [ ] Define shared job kind constants in a package that DB and worker code can use without import cycles.
- [ ] Replace repeated job kind string literals in production code.
- [ ] Extract worker handlers for embed image, annotate image, annotate video, and transcribe video.
- [ ] Existing worker integration tests pass after extraction without weakening assertions.
- [ ] Add coverage that each extracted handler returns the expected success/failure signals used by the queue.
- [ ] Keep claim/fail/retry/complete behavior equivalent to the current public queue path.

## Related Files

- `internal/worker/queue.go`
- `internal/worker/queue_test.go`
- `internal/db/index_jobs.go`
- `internal/images/http.go`
- `internal/videos/http.go`
- `internal/jobs/http.go`
