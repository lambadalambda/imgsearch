# 043: Tighten API Path, Method, and Dependency Semantics

## Priority

P2

## Status

Completed.

## Summary

API handlers should reject unsupported item/action paths consistently, set method metadata consistently, and fail safely when required dependencies are missing.

## Context

- `internal/images/http.go:131` returns the image collection for any GET path handled by the images handler, including item/action paths.
- `internal/videos/http.go:188` has the same collection GET behavior for videos.
- Method-not-allowed responses usually omit `Allow`, while the web UI handler sets it.
- Some handlers validate nil dependencies up front, while image/video/search/upload handlers rely on later calls and can produce inconsistent failures.

## Risks

- Clients can receive successful collection responses for invalid item URLs.
- API behavior differs between packages for the same method/path class.
- Missing dependency failures can become panics or generic 500s depending on handler path.

## Acceptance Criteria

- [x] Require exact collection paths for image and video list GETs.
- [x] Return `404` or `405` consistently for unsupported item/action paths.
- [x] Add a shared method-not-allowed helper that sets `Allow`.
- [x] Standardize constructor or request-time nil dependency validation across handlers.
- [x] Add tests for invalid image/video item GETs and method metadata.

## Resolution

- Added `httputil.WriteMethodNotAllowed`, which returns JSON errors with an `Allow` header.
- Made image/video collection GETs require `/api/images` and `/api/videos` exactly, so item GETs no longer return collection responses.
- Standardized method-first handler checks, with missing dependencies returning `503 Service Unavailable` across API handlers.
- Added regression coverage for invalid item GETs, method metadata, dependency failures, and server mux expectations.

## Related Files

- `internal/images/http.go`
- `internal/images/http_test.go`
- `internal/videos/http.go`
- `internal/videos/http_test.go`
- `internal/httputil/json.go`
- `internal/search/http.go`
- `internal/upload/http.go`
