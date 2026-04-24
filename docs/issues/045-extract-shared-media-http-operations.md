# 045: Extract Shared Image/Video Media Operations

## Priority

P2

## Status

Completed.

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

- [x] Extract shared path-action parsing for item action routes.
- [x] Extract or centralize bool-to-int conversion for SQL query args.
- [x] Consolidate reannotation request logic where the SQL shape can remain explicit and safe.
- [x] Consolidate NSFW tag toggling through a shared helper or small data-layer function.
- [x] Keep delete operations transactional and remove files only after commit.
- [x] Preserve existing image and video API behavior with regression tests.

## Resolution

- Added `httputil.ParseItemActionIDPath` for shared item action route parsing.
- Added `httputil.BoolToInt` and routed image, video, and search SQL helpers through it.
- Added `tagutil.ToggleTagJSON` so image/video NSFW toggles share decode/toggle/encode behavior.
- Added `internal/mediaops.RequestReannotationJob` to share annotation job insert/reset SQL while leaving media-specific row updates explicit.
- Left image/video delete flows transactional with file removal after commit, and kept existing regression coverage green.

## Related Files

- `internal/images/http.go`
- `internal/images/http_test.go`
- `internal/videos/http.go`
- `internal/videos/http_test.go`
- `internal/httputil/`
- `internal/tagutil/`
