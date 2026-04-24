# 036: Harden Batch Job Leasing and Worker Option Defaults

## Priority

P1

## Status

Completed.

## Summary

The batched worker claim path should guarantee that every returned job was actually leased by the requesting worker, and worker runtime defaults should be applied before any claim writes occur.

## Context

- `internal/worker/queue.go:164` selects candidate IDs, updates them, commits, then reloads jobs by ID only.
- `internal/worker/queue.go:257` executes a multi-statement update without checking per-job `RowsAffected`.
- `internal/worker/queue.go:280` reloads by `id IN (...)` without checking `state = 'leased'` and `lease_owner = ?`.
- `internal/worker/queue.go:319` calls `claimBatch` before `ProcessBatch` defaults an empty owner to `worker`.
- `claimNext` already has the stronger single-job pattern at `internal/worker/queue.go:146`.

## Risks

- A batch worker can process jobs that it did not successfully lease if the selected candidates changed before update.
- Empty lease owners can be persisted through the batch path.
- Single-job and batch-worker semantics can drift, making queue bugs hard to reproduce.

## Acceptance Criteria

- [x] Apply owner and lease-duration defaults before batch claiming.
- [x] Ensure `claimBatch` returns only jobs with `state = 'leased'` and `lease_owner` matching the claimant.
- [x] Verify `claimBatch` reloads jobs with `WHERE id IN (...) AND state = 'leased' AND lease_owner = ?` so jobs claimed by another concurrent worker are excluded.
- [x] Check the update result enough to distinguish no successfully claimed jobs from a successful batch claim.
- [x] Remove unused placeholder/query-building code in `claimBatch`.
- [x] Add regression tests for empty owner defaulting and partial/unclaimed batch candidates.

## Related Files

- `internal/worker/queue.go`
- `internal/worker/queue_test.go`
