# 031: Deduplicate Tag JSON Decoding Helpers

## Priority

P2

## Status

Open.

## Summary

`decodeTagsJSON` is currently duplicated across API and worker packages with identical behavior.

## Context

Duplicate copies exist in:

- `internal/images/http.go`
- `internal/videos/http.go`
- `internal/search/http.go`
- `internal/worker/queue.go`

## Risks

- Logic drift and inconsistent error handling.
- Extra maintenance overhead.

## Acceptance Criteria

- [ ] Move tag JSON decoding to a single shared helper.
- [ ] Replace package-local duplicates with shared usage.
- [ ] Keep behavior for empty values and malformed JSON consistent.
- [ ] Add tests for shared helper edge cases.

## Related Files

- `internal/images/http.go`
- `internal/videos/http.go`
- `internal/search/http.go`
- `internal/worker/queue.go`
