# 045: Extract Shared Image/Video Media Operations

## Priority

P2

## Status

Open.

## Summary

Image and video handlers implement similar reannotation, NSFW toggling, delete cleanup, path parsing, and boolean SQL helpers. Extract shared media-level helpers once route semantics are tightened.

This should follow `docs/issues/043-tighten-api-route-semantics.md` so shared path helpers encode the final route behavior instead of preserving today's ambiguous item GET semantics.

## Context

- Reannotation path parsing is duplicated in `internal/images/http.go:207` and `internal/videos/http.go:261`.
- Reannotation state/job reset flows are similar in image and video handlers.
- NSFW toggle logic is duplicated across image and video handlers.
- Delete cleanup patterns both remove database rows and stored paths after transaction commit.
- `boolToInt` exists in multiple packages.

## Risks

- Image and video behavior drifts as features are added to one handler first.
- Future safety fixes for deletion or tag mutation must be repeated.
- Handler files stay large and hard to review.

## Acceptance Criteria

- [ ] Extract shared path-action parsing for item action routes.
- [ ] Extract or centralize bool-to-int conversion for SQL query args.
- [ ] Consolidate reannotation request logic where the SQL shape can remain explicit and safe.
- [ ] Consolidate NSFW tag toggling through a shared helper or small data-layer function.
- [ ] Keep delete operations transactional and remove files only after commit.
- [ ] Preserve existing image and video API behavior with regression tests.

## Related Files

- `internal/images/http.go`
- `internal/images/http_test.go`
- `internal/videos/http.go`
- `internal/videos/http_test.go`
- `internal/httputil/`
- `internal/tagutil/`
