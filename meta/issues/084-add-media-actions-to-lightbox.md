# Expose media actions (NSFW / re-annotate / delete) in the lightbox

## Summary

Flag NSFW, Re-annotate, and Delete exist only in the small hover overflow menu on cards. The lightbox — where the user is inspecting the item most closely — offers only "Find similar" and "Open original". On touch devices the card menu is the harder surface to hit.

## Requirements

- The lightbox action row includes the same actions as the card menu (Flag/Unflag NSFW, Re-annotate, Delete with confirmation).
- Delete closes the lightbox and removes the pin from the masonry (same optimistic behavior as the card path).
- Action errors surface inside the lightbox.

## Acceptance Criteria

- All three actions reachable and functional from the lightbox; covered by smoke test (reuse existing stub endpoints).
- NSFW state change reflects on the underlying card after closing.

## Notes

- `frontend/src/components/Lightbox.svelte:146-165`; action logic currently lives in `Pin.svelte:39-92` — extract to a shared helper rather than duplicating.
- Found during the 2026-06-10 UI/UX review.
