# Compact mobile header and quick-row chrome

## Summary

On a 375×812 viewport, the first image starts ~530px down the page: the header stacks Media / Sort / NSFW / Upload controls vertically (one full-width row each) and the quick-row tag chips wrap onto up to four lines. Roughly two-thirds of the first mobile screen is chrome before any content.

## Requirements

- Header controls fit in at most one or two compact rows on phone widths.
- The quick-row stays a single horizontally scrollable line on mobile (it already does this on desktop via `overflow-x-auto`), instead of wrapping.
- First library pin is visible above the fold on a 375×812 viewport with a populated library.

## Acceptance Criteria

- Mobile-viewport smoke check: the first `[data-pin]` bounding box starts within the initial viewport height.
- No horizontal page overflow introduced at 320–430px widths.

## Notes

- `frontend/src/components/Header.svelte` (control stack), `frontend/src/components/QuickRow.svelte:15` (`flex-wrap sm:flex-nowrap`).
- Found during the 2026-06-10 UI/UX review (mobile screenshot).
