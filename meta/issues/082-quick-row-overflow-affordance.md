# Give the quick-row a visible overflow affordance on desktop

## Summary

The quick-row of tag chips is `overflow-x-auto` with the scrollbar hidden (`scroll-x-soft`). When chips overflow, the last visible chip is clipped mid-letter and nothing indicates more content exists; mouse users have no scrollbar and vertical wheel scrolling doesn't move the row.

## Requirements

- Provide a visible affordance when the row overflows: edge fade, scroll buttons, or simply fitting whole chips (hiding partially-clipped ones).
- Keep the row keyboard- and wheel-reachable (e.g. translate vertical wheel to horizontal scroll, or show the scrollbar on hover).

## Acceptance Criteria

- At a width where chips overflow, no chip renders partially clipped without an overflow indicator.
- All chips remain reachable with a mouse alone.

## Notes

- `frontend/src/components/QuickRow.svelte`, `scroll-x-soft` utility in `frontend/src/app.css:88-96`.
- Found during the 2026-06-10 UI/UX review (visible in README screenshot too: "Tag · m").
