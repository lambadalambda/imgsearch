# Make search modes and scope explicit

## Priority

P2 — Design recommendation.

## Summary

The tag pills look like refinements, but clicking one replaces the search. Counts and bare percentages add further ambiguity.

## Evidence

Twice in production: search “a quiet forest at dawn” → 48 results → click Tag · outdoor → the query clears, heading becomes outdoor, and URL becomes ?tag=outdoor. Media/sort controls disappear in search. Results show bare 42%/39% badges. The library header's image/video totals and visible-result count have different, unexplained scopes.

Reproduction / inspection steps:

1. Search for a quiet forest at dawn.
2. Click Tag · outdoor under the search box.
3. Observe that this replaces the query rather than refining it.

## Risks

A control visually attached to a search box is likely to be read as a filter on that search. “42%” can be mistaken for confidence or probability. Global totals beside the query can look like result totals.

## Requirements

- Choose a clear model: additive tag filters with removable chips, or explicitly labelled Browse tag shortcuts separate from the search controls.
- Preserve the query and filter scope visibly; make media filtering consistent between browsing and searching where supported.
- Label similarity as a relative match score, or replace the raw percentage with rank/relevance presentation. Do not present it as confidence.
- Distinguish Library totals from Visible results; use media/image/card consistently instead of pins and Quick collections for tag shortcuts.
- Provide a visible Search by image button that opens a picker, especially for mobile and keyboard users; paste/drop can remain shortcuts.

## Acceptance Criteria

- [ ] Users can predict whether a tag click refines or replaces a search; counts and match indicators identify their meaning and scope.
- [ ] Add red→green coverage for behavior changes; verify desktop and mobile UI.

## Notes

- Audit only; no fix implemented.
- [HTML report, UX-04](../../docs/reviews/2026-09-22-atelier-ux-audit.html#UX-04) contains embedded screenshot evidence.
