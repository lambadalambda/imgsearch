# 038: Include Videos in Live Updates and Stale-Request Guards

## Priority

P2

## Status

Done.

## Summary

Live snapshots update images and stats, but not videos. Once a live snapshot arrives, polling stops, so video state can go stale while the app reports live updates as active.

## Context

- `internal/live/http.go:36` defines snapshots with images and stats only.
- `internal/live/http.go:155` builds snapshots through `images.List` and `stats.Collect` only.
- `internal/webui/static/app.js:1351` applies live snapshots only to stats and gallery images.
- `internal/webui/static/app.js:1383` stops polling fallback after the first snapshot.
- `internal/webui/static/app.js:1918` skips manual image/video/stat refresh after upload when the live socket is active.
- `loadVideos` in `internal/webui/static/app.js:1176` lacks the stale-request token pattern used by images and stats.

## Risks

- Uploaded videos may not appear until a manual refresh.
- Video indexing, annotation, and transcription states can remain stale while the connection badge says live updates are active.
- Rapid video pagination or refresh can allow older responses to overwrite newer state.

## Acceptance Criteria

- [x] Extend live snapshots to include video list data, or keep a video polling path active while live snapshots remain image-only.
- [x] Add stale-request protection to `loadVideos`, mirroring images and stats.
- [x] Ensure upload refresh behavior updates both image and video state when live mode is active.
- [x] Add regression coverage for live updates with videos.

## Notes

- Live snapshots now include `videos.ListResponse` alongside images and stats.
- The UI applies live video snapshots on the first videos page and re-renders when video state, transcript, description, tags, preview, or storage fields change.
- `loadVideos` already uses the shared `loadMediaCollection` stale-request token path introduced by the frontend collection helper refactor.

## Related Files

- `internal/live/http.go`
- `internal/live/http_test.go`
- `internal/webui/static/app.js`
- `internal/videos/http.go`
- `internal/videos/http_test.go`
