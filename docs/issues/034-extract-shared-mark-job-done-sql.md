# 034: Extract Shared `mark job done` SQL Helper

## Priority

P3

## Status

Open.

## Summary

Job-completion SQL is repeated in multiple worker completion methods and should be centralized.

## Context

Equivalent `UPDATE index_jobs SET state = 'done' ...` statements appear in:

- `completeJob`
- `completeVideoAnnotationJob`
- `completeVideoTranscriptJob`

all under `internal/worker/queue.go`.

## Risks

- Future schema/state changes may be applied inconsistently.
- Avoidable maintenance overhead in a hot path.

## Acceptance Criteria

- [ ] Introduce a shared helper for marking a job done inside a transaction.
- [ ] Replace duplicated SQL blocks with helper calls.
- [ ] Preserve current transactional behavior and error handling.
- [ ] Keep worker tests passing without behavior regressions.

## Related Files

- `internal/worker/queue.go`
- `internal/worker/queue_test.go`
