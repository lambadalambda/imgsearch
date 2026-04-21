# 026: Harden Multipart Upload Resource Usage

## Priority

P1

## Status

Completed.

## Summary

Multipart upload handling currently uses a very large parse threshold and does not explicitly clean parser temp files.

## Context

- `internal/upload/http.go`:
  - `maxUploadBytes = 500 << 20`
  - `r.ParseMultipartForm(maxUploadBytes)`
  - no `r.MultipartForm.RemoveAll()` cleanup

## Risks

- Elevated memory and disk pressure from large or repeated uploads.
- Potential disk fill from lingering multipart temp files.

## Acceptance Criteria

- [x] Separate total request size from multipart in-memory parse threshold.
- [x] Ensure multipart temp files are cleaned up after request handling.
- [x] Enforce a cap on files-per-request.
- [x] Add tests for oversized payload and cleanup behavior.

## Related Files

- `internal/upload/http.go`
- `internal/upload/http_test.go`
