# Cleanup should preserve the places an image was saved

## Priority

P1 — Reproduced workflow issue.

## Summary

“Keep largest” deletes the smaller copies but does not move their collection memberships to the survivor. A tidy-up can quietly empty a saved collection.

## Evidence

In the disposable sandbox, Audit scratchpad contained Cat duplicate variant B. Keeping the largest image in that duplicate group left Audit scratchpad empty. Repeated with a second group (Square companion / Dog portrait) with the same result. The confirmation names dimensions and deletion count, but does not mention affected collections.

Reproduction / inspection steps:

1. Save a smaller duplicate into a collection; confirm it appears there.
2. Open Duplicates, choose Keep largest for that group, and confirm deletion.
3. Return to the collection: the saved image is gone and the surviving copy is not in its place.

## Risks

People think of a duplicate cleanup as keeping the same picture in a better form—not removing it from their organization system. The existing warning says deletion is irreversible, but not what else disappears.

## Requirements

- Make Keep largest a consolidation operation: union collection memberships onto the keeper before deleting copies, in one transaction.
- Related improvement: preserve manual metadata deliberately; surface conflicts instead of silently choosing a winner.
- Preview the keeper, number of removals, and affected collections. State whether files will be deleted or retained.

## Acceptance Criteria

- [ ] A collection containing any removed duplicate still contains the keeper after cleanup, with deterministic metadata handling and rollback on failure.
- [ ] Add red→green coverage for behavior changes; verify desktop and mobile UI.

## Notes

- Audit only; no fix implemented.
- [HTML report, UX-01](../../docs/reviews/2026-09-22-atelier-ux-audit.html#UX-01) contains embedded screenshot evidence.
