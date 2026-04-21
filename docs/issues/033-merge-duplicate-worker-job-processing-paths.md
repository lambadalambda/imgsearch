# 033: Merge Duplicate Worker Job Processing Paths

## Priority

P2

## Status

Completed.

## Summary

`ProcessOne` and `processJob` in the worker queue duplicate most job execution logic.

## Context

- `internal/worker/queue.go` has two large, near-identical flows:
  - `ProcessOne`
  - `processJob`

## Risks

- Bug fixes must be applied twice.
- Behavior drift between single-job and batched worker paths.

## Acceptance Criteria

- [x] Collapse job execution into a single shared implementation.
- [x] Keep existing semantics for claim/fail/retry/complete behavior.
- [x] Preserve logging and metrics expectations.
- [x] Update tests to cover both single and batched execution paths through the shared logic.

## Related Files

- `internal/worker/queue.go`
- `internal/worker/queue_test.go`
