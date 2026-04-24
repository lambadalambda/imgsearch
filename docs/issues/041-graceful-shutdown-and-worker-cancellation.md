# 041: Add Graceful Shutdown and Cancellation-Aware Worker Loops

## Priority

P1

## Status

Open.

## Summary

The process should react to OS signals, stop accepting HTTP requests, cancel workers, and let in-flight operations unwind within a bounded timeout.

## Context

- `cmd/imgsearch/main.go:407` starts the worker with `context.Background()`.
- `cmd/imgsearch/main.go:427` calls `ListenAndServe` directly.
- `internal/worker/runner.go:32`, `36`, `42`, and `46` use `time.Sleep`, so cancellation waits until the sleep ends.
- Long native and ONNX calls only check context at coarse boundaries.

## Risks

- Ctrl-C or service manager stop can leave HTTP requests, workers, or database activity in an abrupt state.
- Worker-only mode cannot be stopped cleanly through a root context.
- Idle worker loops delay shutdown by the full idle sleep.

## Acceptance Criteria

- [ ] Create a root context from OS signals in `cmd/imgsearch`.
- [ ] Pass the root context into worker loops and startup-time operations where appropriate.
- [ ] Use `http.Server.Shutdown` with a bounded timeout for API mode.
- [ ] Replace worker idle sleeps with cancellation-aware `select` logic.
- [ ] Add tests for worker loop cancellation behavior.

## Related Files

- `cmd/imgsearch/main.go`
- `internal/worker/runner.go`
- `internal/worker/queue.go`
- `cmd/imgsearch/main_http_test.go`
