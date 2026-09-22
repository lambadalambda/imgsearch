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
- Implementation: lazy authenticated `/media/images/thumbnails/{id}.webp` generation; 640px max-side WebP at quality 80; shared libvips resizing, atomic file publication, existing `thumbnail_path` persistence/cleanup. API reads advertise paths without generating or writing. Old libraries backfill on first view. See ADR-008 in `docs/decisions.md`.
- Atelier library, image search, and duplicate views use derivatives, retaining originals for the lightbox/Open original and as a one-time error fallback. Video previews and legacy UI are unchanged.
- TDD: missing image/search thumbnail fields and frontend URL mappings failed first; added coverage for real generation, orientation/transparency, no upscaling, cache reuse/regeneration, concurrent requests, deletion during generation, partial-file cleanup, authentication/path containment, and stale files after a database reset. Browser regression checks grid and lightbox requests separately.
- Review: independent correctness/architecture review found that ID-only URLs must revalidate after database resets. Added a failing HTTP-cache regression and switched to private revalidation with subsecond file validators; the follow-up review approved the fix.
- Verification: frontend build/check (zero diagnostics), `mise run test:full` (including 31 frontend unit tests and both browser suites), `go vet ./...`, and race tests for thumbnails/imageutil/images/search/app passed. Real fixture integration preserved the 85,186-byte original and served a 25,808-byte derivative (~70% fewer bytes); this is a single-image check, not a large-library benchmark.
