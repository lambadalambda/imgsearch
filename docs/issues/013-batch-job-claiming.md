# 013: Batch Job Claiming at the Queue Level

## Context

The worker currently claims one job at a time via `claimNext()` (`internal/worker/queue.go:175-255`). Each claim is a SELECT + UPDATE transaction. Between jobs, there's a 500ms idle poll (`internal/worker/runner.go:21`). When processing a batch of images (e.g., after a bulk upload), this means:

1. Claim one job → process it → claim next → 500ms idle if queue appears empty momentarily → repeat
2. Each claim is a SQLite transaction, adding overhead per image
3. When switching between embed and annotate job kinds, models may need to reload if running separately (though currently both are kept resident)

Claiming multiple jobs of the same kind in one transaction would reduce per-image overhead and keep the model "hot" across consecutive jobs without idle gaps.

## Proposed Changes

### A. Batch claim endpoint
Add a `claimBatch(ctx, owner, limit)` method that claims N jobs of the same kind in one transaction:

```sql
SELECT id, kind, image_id, model_id, attempts, max_attempts
FROM index_jobs
WHERE kind = ?
  AND (state = 'pending' OR ...)
ORDER BY created_at ASC
LIMIT ?
```

Then UPDATE all claimed rows in the same transaction.

### B. Batch-aware worker loop
Modify `ProcessOne` to accept a batch of claimed jobs and process them sequentially (or batched, if issue 014 is also done). The worker loop calls `claimBatch` instead of `claimNext`.

### C. Configurable batch size
Add a `-worker-batch-size` flag (default 4-8 for embedding, 1-2 for annotation). Tune based on available VRAM after issue 011 frees up context memory.

### D. Batch DB writes
For embedding jobs, collect all vectors from a batch and upsert them in one transaction instead of one at a time.

## Risks

- Claiming too many jobs at once risks stale leases if the worker crashes mid-batch. The existing lease expiry mechanism handles this (leases expire after 30s), but a batch of 8 embedding jobs at ~200ms each = 1.6s, well within the lease window.
- Batch claiming changes the failure semantics: if job 5 of 8 fails, jobs 6-8 are still leased and will need to wait for lease expiry. This is acceptable given the short processing times.
- SQLite single-connection constraint means batch writes still serialize, but grouping them in one transaction is faster than N separate transactions.

## Acceptance Criteria

- [ ] `claimBatch` method implemented with configurable batch size
- [ ] Worker loop uses batch claiming
- [ ] Queue stats endpoint reflects batch processing accurately
- [ ] Existing job retry and lease expiry still work correctly
- [ ] Reduced idle time between consecutive jobs of the same kind
- [ ] Integration test: bulk upload of 20 images processes faster than before

## Priority

Medium. This is a prerequisite for issue 014 (batched inference) and improves throughput even without it by reducing per-job overhead. Also reduces the 500ms idle gaps between jobs.

## Benchmarking Protocol

### Baseline (before changes)
1. Upload 100 images and record:
   - Total time from first job claimed to last job completed
   - Number of idle gaps (500ms polls between jobs) visible in logs
   - Total DB transaction count (each claim + complete = 2 transactions per job)
2. Record per-job `complete` (DB write) timing from worker logs.

### After changes
1. Same 100-image dataset.
2. Record same metrics plus: batch claim size distribution, time between consecutive jobs of the same kind.
3. Compare total end-to-end time.

### Success criteria
- Fewer total DB transactions (batch claim reduces round-trips)
- Reduced or eliminated idle gaps between same-kind jobs
- Total end-to-end time for 100 images measurably faster

## Estimated Effort

Medium. Changes to `queue.go` (new claim method, batch processing), `runner.go` (loop changes), and `main.go` (new flag). ~100-150 lines changed. Tests for the new claim semantics.
