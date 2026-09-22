# Clarify what “Exact only” means

## Priority

P1 — Design recommendation.

## Summary

The words “Exact only”, “copies”, and “Keep largest” make a perceptual-hash match feel more certain than it is.

## Evidence

The production view offers Exact only and labels groups as copies. In the first captured group, all five images are 450×424, but only the first is labelled largest. There is no visible explanation of ties or quality. The established feature contract is dHash equality/distance—not byte or pixel equality.

Reproduction / inspection steps:

1. Open Duplicates and select Exact only.
2. Inspect a group with equal dimensions and the keeper label.

## Risks

This is a destructive workflow. Hash equality, visual similarity, higher resolution, and better quality are different claims. The current labels blur them together.

## Requirements

- Rename Exact only to Identical visual hash; call the groups Possible matches. Explain that colors, edits, and fine detail may differ.
- Show an explicit Keep badge and let the user select another keeper. For equal dimensions say Same resolution, rather than implying one is larger.
- Add file size/format and an accessible side-by-side comparison. On phones, offer a two-up/flip comparison instead of only a long vertical list.
- Show the separate unattended-import policy/status when that feature is exposed in UI; do not imply the new import option fixes manual cleanup.

## Acceptance Criteria

- [x] No label claims byte equality without verification; users can identify the keeper and understand the deletion/retention policy before confirming.
- [x] Add red→green coverage for behavior changes; verify desktop and mobile UI.

## Notes

- Implemented in the bounded UX worktree; parent review/integration and index/archive/changelog handling remain pending. No commits or deployment from this worktree.
- See [manual consolidation policy/API](../../docs/manual-duplicate-cleanup.md). Labels now say Identical visual hash / Possible matches and explicitly distinguish dHash from byte/pixel equality. Native radio controls choose the keeper, ties say Same resolution, and cards show format plus size when safely available.
- Full-original, uncropped two-up comparison works at phone widths; keyboard-accessible flip and comparison selection are also available. Metadata conflicts receive focus and block confirmation. Preview names the keeper, removals, affected collections and retained-file/no-disk-space policy.
- Red→green evidence: the delivery-build browser regression first failed on `Exact only` versus `Identical visual hash`; frontend unit tests first failed on the missing duplicate helper. A further focused regression failed because conflicts did not receive focus, then passed after the focus fix.
- `scripts/duplicates_ui_test.mjs` passes at 1280×900 and 390×844 (keyboard keeper, ties/format/size, two-up geometry, uncropped images, flip, no horizontal overflow, focused conflicts, cancel, stale-confirm failure, successful consolidation). It is included in root `npm test`/`mise run test:ui`. The existing Atelier smoke duplicate flow now requires preview+confirm POSTs and no per-image DELETEs.
- Current settings API exposes no unattended-import policy/status; this pane clearly separates manual cleanup without inventing a status or changing Settings. Real-device Safari remains unverified.
- [HTML report, UX-02](../../docs/reviews/2026-09-22-atelier-ux-audit.html#UX-02) contains embedded screenshot evidence.

## Integration review

- Independent review found no transactional data-safety blockers. Equal-area/different-dimension candidates now say Same pixel count; identical dimensions say Same resolution, with a pure regression.
- Confirmation retains originals/thumbnails and explicitly says it does not reclaim disk space. Metadata conflicts require resolving edits before confirmation; no silent title winner is chosen. No restore UI or file garbage collection added.
- Inherited concurrent discovery/annotation NSFW classification races and shared confirmation-dialog descriptions are separately tracked. Browser tests use fixtures; real-device Safari and competing WAL/leased-worker/cache races were not directly exercised by this endpoint's tests.

- Parent integrated full suite, frontend check, focused images/collections/worker race tests and formatting passed. Archived for this release.
