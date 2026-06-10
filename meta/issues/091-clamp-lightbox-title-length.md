# Tame sentence-length derived titles in the lightbox

## Summary

`deriveTitle` uses the first sentence of the model-generated description when no explicit title exists. In the mobile lightbox this renders as a huge multi-line 24px serif headline (observed: seven lines for "A screenshot of a digital space simulation or astronomical model set against a black starfield.") that pushes the actual description and actions far below the fold.

## Requirements

- Lightbox title is visually clamped (e.g. 2-3 lines with line-clamp) and/or rendered smaller on phone widths.
- The full text remains available (it already repeats in the description section; avoid duplicate display when title == description first sentence).

## Acceptance Criteria

- On a 375px viewport with a sentence-length title, the title occupies at most ~3 lines and the description/actions are reachable without scrolling past a giant headline.
- Desktop layout unchanged for short titles.

## Notes

- `frontend/src/components/Lightbox.svelte:90-92`, `deriveTitle` in `frontend/src/lib/utils.ts:37-46`.
- A future annotator-produced short title field would fix this at the source; this issue is the display-side mitigation.
- Found during the 2026-06-10 UI/UX review.
