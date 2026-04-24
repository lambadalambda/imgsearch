# 043: Tighten API Path, Method, and Dependency Semantics

## Priority

P2

## Status

Open.

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

- [ ] Require exact collection paths for image and video list GETs.
- [ ] Return `404` or `405` consistently for unsupported item/action paths.
- [ ] Add a shared method-not-allowed helper that sets `Allow`.
- [ ] Standardize constructor or request-time nil dependency validation across handlers.
- [ ] Add tests for invalid image/video item GETs and method metadata.

## Related Files

- `internal/images/http.go`
- `internal/images/http_test.go`
- `internal/videos/http.go`
- `internal/videos/http_test.go`
- `internal/httputil/json.go`
- `internal/search/http.go`
- `internal/upload/http.go`
