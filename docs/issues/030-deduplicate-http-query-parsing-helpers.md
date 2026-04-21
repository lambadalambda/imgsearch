# 030: Deduplicate HTTP Query Parsing Helpers

## Priority

P2

## Status

Completed.

## Summary

`limit`, `offset`, and `include_nsfw` parsing helpers are duplicated across multiple handlers.

## Context

Duplicate helper sets currently exist in:

- `internal/images/http.go`
- `internal/videos/http.go`
- `internal/search/http.go`
- `internal/live/http.go`

## Risks

- Behavior drift between endpoints.
- Repeated bug/security fixes in multiple files.

## Acceptance Criteria

- [x] Introduce shared query parsing helpers in a common package.
- [x] Remove duplicate implementations from handler files.
- [x] Preserve existing endpoint behavior and bounds.
- [x] Add/update tests for shared helper behavior.

## Related Files

- `internal/images/http.go`
- `internal/videos/http.go`
- `internal/search/http.go`
- `internal/live/http.go`
- `internal/httputil/`
- `internal/httputil/query.go`
- `internal/httputil/query_test.go`
