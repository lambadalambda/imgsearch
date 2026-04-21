# 030: Deduplicate HTTP Query Parsing Helpers

## Priority

P2

## Status

Open.

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

- [ ] Introduce shared query parsing helpers in a common package.
- [ ] Remove duplicate implementations from handler files.
- [ ] Preserve existing endpoint behavior and bounds.
- [ ] Add/update tests for shared helper behavior.

## Related Files

- `internal/images/http.go`
- `internal/videos/http.go`
- `internal/search/http.go`
- `internal/live/http.go`
- `internal/httputil/`
