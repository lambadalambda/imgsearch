# 044: Clarify Multipart Batch Upload Semantics

## Priority

P2

## Status

Open.

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

- [ ] Keep partial success semantics and return per-file result/error objects for multipart uploads.
- [ ] Use `207 Multi-Status` for mixed success/failure batches.
- [ ] Detect `http.MaxBytesError` and return `413` for oversized requests.
- [ ] Add tests for mixed valid/invalid multipart uploads.
- [ ] Add/update API docs or README notes if response shape changes.

## Related Files

- `internal/upload/http.go`
- `internal/upload/http_test.go`
- `internal/upload/service.go`
- `README.md`
- `docs/development.md`
