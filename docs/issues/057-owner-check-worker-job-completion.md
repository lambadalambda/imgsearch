# 057: Owner-Check Worker Job Completion and Failure Updates

## Priority

P1

## Status

Open.

## Summary

Worker jobs are leased with `lease_owner`, but completion and failure updates only match by job ID. A worker can finish after its lease expires and overwrite another worker's later claim/result.

## Context

- `internal/worker/queue.go` claims jobs by setting `state = 'leased'`, `lease_owner`, and `leased_until`.
- `markJobDoneTx` updates `index_jobs` with `WHERE id = ?` only.
- `failJob` also updates by `WHERE id = ?` only.
- Long embedding, annotation, or transcription work can exceed the lease and be reclaimed.

## Risks

- Two workers can process the same job after lease expiry.
- A stale worker can mark a job done or failed after another worker has claimed it.
- Generated metadata can be overwritten by an older attempt.

## Acceptance Criteria

- [ ] Add a regression test simulating lease expiry, reclaim by another owner, and stale completion/failure by the first owner.
- [ ] Include owner/token/state checks in completion and failure `WHERE` clauses.
- [ ] Detect zero-row completion/failure updates and handle them as stale claims rather than successful writes.
- [ ] Consider lease renewal for long-running work, or document why owner checks are sufficient.

## Related Files

- `internal/worker/queue.go`
- `internal/worker/queue_test.go`
