# 057: Owner-Check Worker Job Completion and Failure Updates

## Priority

P1

## Status

Resolved (2026-06-05). markJobDoneTx, failOrRetry, and the new Queue.RenewLease helper all check `state = 'leased' AND lease_owner = ? AND leased_until > datetime('now')` and return ErrStaleClaim on zero rows. The worker loop now treats ErrStaleClaim as a non-error so it does not double-fail a reclaimed job. Coverage in TestCompleteJobRejectsStaleClaimByOtherOwner, TestFailOrRetryRejectsStaleClaimByOtherOwner, TestCompleteVideoJobRejectsStaleClaimByOtherOwner, and TestRenewLeaseExtendsActiveOwner.

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

- [x] Add a regression test simulating lease expiry, reclaim by another owner, and stale completion/failure by the first owner.
- [x] Include owner/token/state checks in completion and failure `WHERE` clauses.
- [x] Detect zero-row completion/failure updates and handle them as stale claims rather than successful writes.
- [x] Consider lease renewal for long-running work, or document why owner checks are sufficient (added `Queue.RenewLease` so long-running kinds can extend their lease; default 30s is short enough that the embedded Qwen/Gemma paths usually don't need it, but it is available for the 26B annotator and long transcribes).

## Related Files

- `internal/worker/queue.go`
- `internal/worker/queue_test.go`
