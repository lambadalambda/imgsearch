# Persist view preferences across reloads

## Summary

`includeNSFW`, `librarySort`, and `libraryMedia` are plain in-memory stores — not reflected in the URL or localStorage. Every reload silently resets to Random / All media / NSFW hidden.

## Requirements

- Persist the three preferences in `localStorage` and restore them on boot.
- Invalid/missing stored values fall back to current defaults.

## Acceptance Criteria

- Set Sort to "Recently added", media to "Videos only", NSFW to shown; reload → all three retained.
- Smoke test reloads the page and asserts the persisted state.

## Notes

- `frontend/src/lib/stores.ts:100-104`.
- URL-encoding these was considered; localStorage is the smaller change and matches their "device preference" nature. Revisit URL params if shareable links ever matter for a local-first app.
- Found during the 2026-06-10 UI/UX review.
