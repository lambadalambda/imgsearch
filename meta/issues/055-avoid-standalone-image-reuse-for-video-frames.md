# 055: Avoid Reusing Standalone Images as Video Frames

## Priority

P0

## Status

Open.

## Summary

Video frame storage deduplicates by image SHA and can attach an existing standalone uploaded image row as a video frame. Deleting that video can later delete the standalone image row and file.

## Context

- `internal/upload/service.go` inserts sampled video frames into `images` with `ON CONFLICT(sha256) DO NOTHING`.
- On conflict, it loads the existing `images.id` and inserts a `video_frames` row pointing at it.
- `internal/images/http.go` hides any `images` row referenced by `video_frames` from the standalone image list.
- `internal/videos/http.go` deletes frame image rows when deleting a video if no remaining `video_frames` reference exists.
- Decision: sampled video frames should use separate frame-owned rows and must not attach `video_frames` to standalone image rows, even when bytes are identical.

## Risks

- A user-uploaded standalone image can disappear from the library after a video import samples an identical frame.
- Deleting the video can delete the shared standalone image file and metadata.
- Deduplication ownership is ambiguous between source media and derived video-frame media.

## Acceptance Criteria

- [ ] Add a regression test where a sampled video frame hashes to an existing standalone image.
- [ ] Ensure importing the video does not hide, reclassify, or delete the standalone image.
- [ ] Ensure deleting the video only removes frame-owned derived image rows/files.
- [ ] Capture the chosen ownership model in comments or docs.

## Related Files

- `internal/upload/service.go`
- `internal/images/http.go`
- `internal/videos/http.go`
- `internal/upload/service_test.go`
- `internal/videos/http_test.go`
