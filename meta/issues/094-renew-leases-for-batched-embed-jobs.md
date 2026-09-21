# Renew leases for batched embed jobs

## Summary

`processEmbedJobsBatch` (`internal/worker/queue.go:407-497`) never calls `startLeaseRenewer`, unlike the single-job path at line 506. With the default 30s lease, a CPU batch that takes longer than that fails `markJobDoneTx` with `ErrStaleClaim`, and `failOrRetry` has the same guard so it updates zero rows. The jobs stay `leased` with an expired lease, `claimBatch` reclaims them on the next tick, and the batch is embedded again indefinitely. `attempts` grows but the jobs never reach `failed`.

Only affects `-worker-batch-size > 1`.

## Requirements

- Start one lease renewer covering all jobs in a batch before embedding begins, and stop it after completion or failure is recorded.
- Alternatively or additionally, size the initial batch lease as `now + N * lease` in `claimBatch`.

## Acceptance Criteria

- A queue test with batch size > 1 and an embedder that sleeps past the lease duration completes all jobs as `done` and never re-claims them.
- A failing batch embedder marks the jobs `failed` after `max_attempts` instead of looping.

## Notes

- Existing batch test at `internal/worker/queue_test.go:1546` covers only the happy path.
- Found during the 2026-09-21 review.
