# 065: Preserve Result Rank Across Masonry Rows

## Priority

P1

## Status

Done.

## Summary

Atelier currently uses CSS multi-column layout for masonry results. CSS columns fill down each column before moving across, so similarity/search results are visually ranked by column rather than left-to-right across rows. This makes the first visible row contain less-similar items from deeper in the result list.

## Context

- `frontend/src/components/Masonry.svelte` renders pins in backend rank order into a CSS `columns-*` container.
- In CSS multi-column layout, DOM order flows top-to-bottom in the first column, then the next column.
- For similarity browsing, users expect the most similar results to appear across the first row before lower-ranked results.

## Risks

- The most relevant matches look scattered or missing from the top row.
- Users may misread the ranking quality because top-row cards are not the top-ranked results.
- Future UI tests can pass DOM-order assertions while visual ordering is wrong.

## Acceptance Criteria

- [x] Add a browser regression test that checks the first visible row follows result rank order.
- [x] Replace CSS multi-column flow with explicit rank-aware column distribution.
- [x] Preserve responsive column counts: 1 column on small screens, then 2/3/4 columns at existing breakpoints.
- [x] Preserve card spacing, load-more behavior, and empty/loading states.
- [x] Keep DOM/test hooks stable enough for existing smoke tests.

## Resolution

- Replaced CSS multi-column flow with a direct CSS grid whose children remain in rank/DOM order.
- Each pin is measured with `ResizeObserver` and assigned a grid row span, preserving masonry-style packing while keeping the first visible row as ranks 1 through N from left to right.
- Added an Atelier smoke assertion that a text-search top row renders `95%`, `91%`, `90%`, `86%` in visual order.

## Related Files

- `frontend/src/components/Masonry.svelte`
- `frontend/src/components/Pin.svelte`
- `scripts/ui_smoke_test_atelier.mjs`
