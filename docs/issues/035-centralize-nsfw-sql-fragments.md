# 035: Centralize NSFW SQL Fragments Across List/Search Queries

## Priority

P2

## Status

Open.

## Summary

NSFW filtering SQL fragments are repeated across images, videos, and search query paths.

## Context

The same `json_each(...)= 'nsfw'` checks and video-frame NSFW joins are duplicated in multiple queries in:

- `internal/images/http.go`
- `internal/videos/http.go`
- `internal/search/http.go`

## Risks

- Query behavior drift across endpoints.
- Difficult, error-prone maintenance when NSFW semantics evolve.

## Acceptance Criteria

- [ ] Define shared SQL fragment constants or builders for NSFW checks.
- [ ] Reuse shared NSFW logic in image/video/search count and data queries.
- [ ] Preserve existing filtering semantics (including grouped video behavior).
- [ ] Add regression tests to ensure equivalent filtering across endpoints.

## Related Files

- `internal/images/http.go`
- `internal/videos/http.go`
- `internal/search/http.go`
