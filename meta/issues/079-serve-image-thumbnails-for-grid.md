# Serve thumbnail derivatives for grid images

## Summary

The masonry grid loads full-resolution originals for image pins: `pinFromImage` sets `thumbUrl` to the same `/media/<storage_path>` URL as `mediaUrl`. Videos already have a `preview_path` derivative, images have none. On a large library (10k+ images) the grid pulls megabytes per card, slowing scroll and wasting memory.

## Requirements

- Backend generates (or serves on demand) a bounded-size thumbnail derivative for images (e.g. max side ~640px, WEBP), reusing the existing libvips pipeline.
- API responses expose the thumbnail path alongside `storage_path`.
- Frontend uses the thumbnail in the masonry and keeps the original for the lightbox / "Open original".
- Existing libraries get thumbnails backfilled (lazily on first request is acceptable).

## Acceptance Criteria

- Grid `<img>` requests resolve to thumbnail URLs, not originals, for image pins.
- Lightbox still loads the original.
- Go tests cover thumbnail generation/serving; smoke test stub updated accordingly.

## Notes

- Frontend: `frontend/src/lib/utils.ts:61-79`. Backend: media serving + ingestion pipeline (libvips already a dependency).
- Largest item from the 2026-06-10 UI/UX review; backend work dominates.
