# Use configured remote annotation concurrency in worker

## Summary

The user noticed annotations appear to run one at a time and suggested two in parallel. Read-only production settings inspection confirmed the active backend is OpenAI-compatible and its configured request concurrency is already 2. The worker currently starts one loop and processes annotation jobs serially, so that capacity is not used.

## Context

- `cmd/imgsearch/main.go` launches a single `worker.RunLoopBatch`.
- `internal/worker/runner.go` waits for each `ProcessOne`/`ProcessBatch` call.
- `internal/worker/queue.go` processes non-embedding batches in a sequential loop; `worker-batch-size` is not annotation concurrency.
- `internal/embedder/openaicompat/annotator.go` already enforces a configurable semaphore (default 2).
- User subsequently requested implementation: the Settings parallel-request value must control actual concurrency (e.g. setting 2 really runs two). Reuse the existing field; no production deployment or configuration change requested yet.

## Requirements

- Make remote annotation scheduling honor bounded parallelism (initial target: two in-flight jobs), without parallelizing native inference or altering embedding batch semantics indiscriminately.
- Preserve job ownership, lease renewal, cancellation, retries, and safe annotation-backend settings changes while requests are in flight.
- Avoid multiple processes loading duplicate native embedding/model resources just to achieve remote request concurrency.

## Acceptance Criteria

- A blocking fake remote annotator test proves two jobs overlap, a third waits, and the configured cap is respected.
- Tests cover leases, cancellation, failures, and settings/backend changes without duplicate work or resource closure during an active call.
- Benchmark throughput at one versus two before claiming a speedup; independently review and keep this separate from collections.

## Outcome

- Worker admission now follows saved remote concurrency independently of embedding batch size; Settings labels it **Parallel annotations** and explains 1 vs 2. No new setting/schema or dependency required.
- Live limit/backend changes stop admission and drain calls before swapping resources. Native inference stays serial; embedding/transcription do not overlap annotation jobs. Active job IDs are excluded during explicit requeues; shutdown joins jobs and lease renewers.
- Deterministic blocking-fake tests followed red→green and cover two overlapping jobs/third waiting, live limits, video jobs, cancellation, retries, leases, ownership/requeues, priority draining, and resolver resource lifetime. Review strengthened the lease test to prove the original calls survive renewal without reclaims; its ten repeated race runs passed.
- Settings browser coverage saves 1, reloads, saves 2, and reloads again, checking API payloads and field values. Full Go/frontend/script/browser tests, focused race checks, formatting, frontend typecheck and build pass. Independent correctness/architecture review approved; polling documentation reflects possible delays from non-annotation work.
- No real remote annotation requests, throughput benchmark claims, deployment, or production configuration changes. Limits apply per worker process. The [shared-frame duplicate inference edge case](avoid-duplicate-shared-frame-annotation.md) remains a separate non-blocking follow-up.
