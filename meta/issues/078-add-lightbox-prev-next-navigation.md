# Add prev/next navigation to the lightbox

## Summary

The lightbox shows a single pin with no way to move to the adjacent result; users must close it and click the next card. For a gallery app this is the largest interaction gap — flipping through results is the core loop of reviewing a library or a search result set.

## Requirements

- Left/Right arrow keys move to the previous/next pin in the current results (`pins` store order).
- On-screen prev/next affordances (chevrons) for mouse/touch users, hidden or disabled at the ends of the list.
- Video pins keep working: navigating away stops playback; navigating onto a video behaves like opening it fresh.
- Escape and backdrop-close behavior unchanged.

## Acceptance Criteria

- Open the first pin, press ArrowRight → lightbox shows the second pin without closing.
- Prev is disabled/hidden on the first pin; next on the last.
- Smoke test covers keyboard navigation between two pins.

## Notes

- `frontend/src/components/Lightbox.svelte`; current pin comes from `lightboxPin` store — navigation needs the index within `$pins`.
- "Load more" interaction: reaching the last loaded pin can simply stop; auto-paging is out of scope.
- Found during the 2026-06-10 UI/UX review.
