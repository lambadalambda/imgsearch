# Use configured remote annotation concurrency in worker

## Summary

The user noticed annotations appear to run one at a time and suggested two in parallel. Read-only production settings inspection confirmed the active backend is OpenAI-compatible and its configured request concurrency is already 2. The worker currently starts one loop and processes annotation jobs serially, so that capacity is not used.

## Context

- `cmd/imgsearch/main.go` launches a single `worker.RunLoopBatch`.
- `internal/worker/runner.go` waits for each `ProcessOne`/`ProcessBatch` call.
- `internal/worker/queue.go` processes non-embedding batches in a sequential loop; `worker-batch-size` is not annotation concurrency.
- `internal/embedder/openaicompat/annotator.go` already enforces a configurable semaphore (default 2).
- This is a finding/proposed follow-up, not authorization to change production configuration or deploy.

## Requirements

- Make remote annotation scheduling honor bounded parallelism (initial target: two in-flight jobs), without parallelizing native inference or altering embedding batch semantics indiscriminately.
- Preserve job ownership, lease renewal, cancellation, retries, and safe annotation-backend settings changes while requests are in flight.
- Avoid multiple processes loading duplicate native embedding/model resources just to achieve remote request concurrency.

## Acceptance Criteria

- A blocking fake remote annotator test proves two jobs overlap, a third waits, and the configured cap is respected.
- Tests cover leases, cancellation, failures, and settings/backend changes without duplicate work or resource closure during an active call.
- Benchmark throughput at one versus two before claiming a speedup; independently review and keep this separate from collections.
