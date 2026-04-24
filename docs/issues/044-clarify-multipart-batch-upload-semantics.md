# 044: Clarify Multipart Batch Upload Semantics

## Priority

P2

## Status

Completed.

## Summary

Batch upload currently stores files one by one and returns a single error if a later file fails. The API should make partial success explicit or make the operation atomic.

## Context

- `internal/upload/http.go:60` loops over uploaded files and stores each file immediately.
- If file N succeeds and file N+1 is unsupported or fails, the handler returns an error while earlier files remain stored.
- `internal/upload/http.go:38` uses `http.MaxBytesReader`, but parse failures are collapsed into `400 invalid multipart upload` instead of distinguishing payload-too-large.

## Risks

- Clients cannot tell which files succeeded after a mixed success/failure batch.
- Retrying a failed batch can create duplicate handling surprises.
- Oversized upload errors are less actionable than a `413 Payload Too Large` response.

## Acceptance Criteria

- [x] Keep partial success semantics and return per-file result/error objects for multipart uploads.
- [x] Use `207 Multi-Status` for mixed success/failure batches.
- [x] Detect `http.MaxBytesError` and return `413` for oversized requests.
- [x] Add tests for mixed valid/invalid multipart uploads.
- [x] Add/update API docs or README notes if response shape changes.

## Resolution

- Upload responses now include one `uploads[]` entry per submitted file, with either success data or `filename` plus `error`.
- Mixed success/failure batches return `207 Multi-Status`; all-failed file batches return `400` with per-file errors; request-level validation errors remain top-level JSON errors.
- Oversized multipart requests now return `413 Payload Too Large` when `http.MaxBytesError` is detected.
- The web UI surfaces mixed-batch failures, keeps the upload modal open for failed files, and still refreshes successful uploads.
- README notes the multipart response behavior for API clients.

## Related Files

- `internal/upload/http.go`
- `internal/upload/http_test.go`
- `internal/upload/service.go`
- `README.md`
- `docs/development.md`
