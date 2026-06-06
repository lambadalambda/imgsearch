# 067: Renew Worker Leases During Long Jobs

## Priority

P1

## Status

Resolved (2026-06-06). Slow CPU-only annotation jobs now renew their job lease while running, so completion is not rejected as a stale claim after the default 30-second lease expires.

## Summary

CPU-only image and video annotation can exceed the worker lease duration. With owner-checked completion, these long jobs could finish successfully but fail to mark the job done because `leased_until` had already expired.

## Context

- Deployment logs showed repeated `worker: stale job claim` messages for the same annotation jobs.
- The same jobs were reclaimed immediately, causing attempts to grow far beyond `max_attempts` while generated annotations were discarded.
- `Queue.RenewLease` existed but long-running worker execution did not call it.

## Acceptance Criteria

- [x] Add regression coverage for an annotation job that runs longer than its initial lease.
- [x] Renew active worker leases while a claimed job is running.
- [x] Preserve stale-claim protection when a lease is actually lost or reclaimed.
- [x] Verify the deployed worker can complete annotation jobs again.

## Related Files

- `internal/worker/queue.go`
- `internal/worker/queue_test.go`
