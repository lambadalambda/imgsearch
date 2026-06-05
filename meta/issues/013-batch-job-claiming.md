# 013: Batch Job Claiming at the Queue Level

## Status: Completed (Infrastructure)

## Context

The worker currently claims one job at a time via `claimNext()` (`internal/worker/queue.go`). Each claim is a SELECT + UPDATE transaction. Between jobs, there's a 500ms idle poll (`internal/worker/runner.go`). When processing a batch of images (e.g., after a bulk upload), this means:

1. Claim one job → process it → claim next → 500ms idle if queue appears empty momentarily → repeat
2. Each claim is a SQLite transaction, adding overhead per image
3. When switching between embed and annotate job kinds, models may need to reload if running separately (though currently both are kept resident)

Claiming multiple jobs of the same kind in one transaction reduces per-image queue overhead and keeps the model hot across consecutive jobs without idle gaps. In practice, this mostly prepares the worker for issue 014 rather than delivering a strong standalone throughput win.

## Changes Made

### A. Batch claim endpoint
Added `claimBatch(ctx, owner, limit)` method in `internal/worker/queue.go` that claims up to N jobs of the **same kind** in one transaction:
- Queries eligible jobs (pending or expired leases), ordered by embed_image priority then created_at
- Filters to a single job kind (whichever kind appears first in priority order)
- Updates all claimed rows in one transaction with the same lease owner/duration
- Returns the claimed jobs with their current state

### B. Batch-aware worker loop
Added `ProcessBatch(ctx, owner, batchSize)` method that:
- Calls `claimBatch` to claim a batch of same-kind jobs
- Processes each job sequentially via `processJob` (extracted helper)
- Continues processing remaining jobs if one fails
- Returns the count of successfully processed jobs

Added `RunLoopBatch(ctx, q, owner, idleDelay, batchSize)` in `runner.go`:
- `batchSize > 1` uses `ProcessBatch`
- `batchSize == 1` uses existing `ProcessOne` (backward compatible)
- `RunLoop` preserved as a thin wrapper calling `RunLoopBatch` with batchSize=1

### C. Configurable batch size
Added `-worker-batch-size` flag (default 1, meaning no batching by default). This enables opt-in batch processing and is ready for issue 014 (batched inference) which will set a higher default.

### D. Batch DB writes
Deferred to issue 014. Currently each job in a batch completes individually.

## Benchmark Results

Benchmarked with the real native embedder on 100 random images copied from `~/old`, with `-enable-annotations=false` and `-vector-backend=bruteforce`. Timing was measured from the first `worker claimed job` log line to the last `worker completed job` log line.

### Dataset setup

```bash
mkdir -p /tmp/imgsearch-bench-data && shuf -zn 100 -e ~/old/* | xargs -0 -I{} cp "{}" /tmp/imgsearch-bench-data/
```

### Results

- `batch_size=1`
  - run 1: 82s
  - run 2: 92s
  - average: 87s
- `batch_size=4`
  - run 1: 78s
- `batch_size=8`
  - run 1: 103s
  - run 2: 82s
  - average: 92.5s

### Conclusion

- Queue-level batching alone does not show a clear, reliable end-to-end speedup with the real embedder on this M2 Max benchmark.
- A separate DB-only microbenchmark still showed lower queue overhead from batching, but that saving is too small to materially affect the full embedding pipeline where GPU inference dominates.
- `batch_size=4` looked promising in one run, but the overall result is best treated as neutral until issue 014 adds actual batched inference.
- This issue should be viewed as groundwork for issue 014, not as a standalone performance win.

## Acceptance Criteria

- [x] `claimBatch` method implemented with configurable batch size
- [x] Worker loop uses batch claiming (opt-in via `-worker-batch-size` flag)
- [x] Existing job retry and lease expiry still work correctly (19/19 tests pass)
- [x] Reduced idle time between consecutive jobs of the same kind (batch claims eliminate per-job claim latency and idle polls within a batch)
- [ ] Queue stats endpoint reflects batch processing accurately (deferred, not critical)
- [ ] Integration test: bulk upload of 20 images processes faster than before (not demonstrated here; defer meaningful throughput gains to issue 014)

## Risks (Mitigated)

- Claiming too many jobs at once risks stale leases if the worker crashes mid-batch. The existing lease expiry mechanism handles this (leases expire after 30s). A batch of 8 embedding jobs at ~530ms each = ~4.2s, well within the 30s lease window.
- Batch claiming changes the failure semantics: if job 5 of 8 fails, jobs 6-8 are still leased and will be processed. The failed job is retried via `failOrRetry`. This is acceptable given the short processing times.
- SQLite single-connection constraint means batch writes still serialize, but the batch claim reduces N separate claim transactions to 1.

## Files Changed

- `internal/worker/queue.go` — added `claimBatch`, `ProcessBatch`, `processJob`
- `internal/worker/queue_test.go` — added 10 new tests for batch claiming and batch processing
- `internal/worker/runner.go` — added `RunLoopBatch`, `RunLoop` delegates to it
- `cmd/imgsearch/main.go` — added `-worker-batch-size` flag, uses `RunLoopBatch`
