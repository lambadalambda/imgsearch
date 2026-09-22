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

- [x] A collection containing any removed duplicate still contains the keeper after cleanup, with deterministic metadata handling and rollback on failure.
- [x] Add red→green coverage for behavior changes; verify desktop and mobile UI.

## Notes

- Implemented in the bounded UX worktree; parent review/integration and index/archive/changelog handling remain pending. No commits or deployment from this worktree.
- See [manual consolidation policy/API](../../docs/manual-duplicate-cleanup.md). Cleanup now previews and consolidates onto a user-selected keeper in one transaction. Original files/thumbnails are deliberately retained (no disk space reclaimed); the confirmation states this.
- Collection union retains the keeper's membership/order when present, otherwise the earliest removed membership. Compatible manual titles/tag provenance are preserved deterministically; conflicting titles or tag add/remove intent block confirmation. Unattended-import semantics are unchanged.
- Red→green evidence: new `TestDuplicateConsolidation*` requests initially failed with 405; after implementation, collection union and injected transfer/deletion failure tests pass, including exact membership rows, title/tags/removals, embeddings and jobs under foreign keys. Stale preview, invalid group/ID, standalone/transitive/NSFW and retained-file checks pass.
- File-size regression initially failed with a missing size; confined filesystem lookup now returns the safe file size and refuses symlink/absolute-path escapes.
- Browser red→green is recorded in UX-02. API browser fixtures and SQLite tests are separate; real-device Safari/model-dependent native integration were not exercised.
- [HTML report, UX-01](../../docs/reviews/2026-09-22-atelier-ux-audit.html#UX-01) contains embedded screenshot evidence.

## Integration review

- Independent review found no transactional data-safety blockers. Equal-area/different-dimension candidates now say Same pixel count; identical dimensions say Same resolution, with a pure regression.
- Confirmation retains originals/thumbnails and explicitly says it does not reclaim disk space. Metadata conflicts require resolving edits before confirmation; no silent title winner is chosen. No restore UI or file garbage collection added.
- Inherited concurrent discovery/annotation NSFW classification races and shared confirmation-dialog descriptions are separately tracked. Browser tests use fixtures; real-device Safari and competing WAL/leased-worker/cache races were not directly exercised by this endpoint's tests.

- Parent integrated full suite, frontend check, focused images/collections/worker race tests and formatting passed. Archived for this release.
