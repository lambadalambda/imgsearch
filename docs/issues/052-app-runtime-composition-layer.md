# 052: Promote internal/app Into the Runtime Composition Layer

## Priority

P2

## Status

Open.

## Summary

Move reusable DB/vector/model/worker/HTTP construction behind `internal/app` boundaries after runtime config and graceful shutdown are in place.

## Context

- `cmd/imgsearch/main.go` handles SQLite/vector setup, migrations, default model downloads, embedder/annotator/transcriber construction, queue construction, and HTTP server setup.
- `internal/app/bootstrap.go` currently only applies DB pragmas, runs migrations, and validates the vector backend.
- `docs/issues/048-runtime-config-extraction.md` should extract typed runtime configuration first.
- `docs/issues/041-graceful-shutdown-and-worker-cancellation.md` should land first so lifecycle ownership is clear before moving construction.

## Risks

- Startup behavior is hard to unit test without invoking command-level wiring.
- API and worker modes can drift as more conditional setup is added.
- Model/runtime lifecycle cleanup remains scattered through `main` defers.
- Starting this too early will conflict with shutdown and worker refactors.

## Acceptance Criteria

- [ ] Move DB/vector initialization behind an `internal/app` function where practical.
- [ ] Move model runtime construction behind small factories with clear lifecycle ownership.
- [ ] Return an app/runtime object that owns HTTP mux, queue, model closers, and shutdown hooks.
- [ ] Keep `cmd/imgsearch` focused on flags, logging, signal handling, and process exit.
- [ ] Preserve API, worker, and all-in-one mode behavior with tests.

## Related Files

- `cmd/imgsearch/main.go`
- `cmd/imgsearch/*_test.go`
- `internal/app/bootstrap.go`
- `internal/app/bootstrap_test.go`
- `internal/vectorindex/`
- `internal/embedder/`
- `internal/worker/`
