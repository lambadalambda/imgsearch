# Make Collections feel like saved media

## Priority

P2 — Design recommendation.

## Summary

The collection index resembles a settings list: names and management buttons, but no visual memory cues or counts.

## Evidence

Populated and empty collections look alike in the list. Rename/Delete occupy much of each row. The empty-detail state says items might be NSFW-hidden even for a genuinely empty list, and offers no direct Browse library action. Create, save, rename and delete worked in the sandbox.

Reproduction / inspection steps:

1. Open Collections with one populated and one empty list.
2. Open the empty list and inspect its next-step guidance.

## Risks

Collections are a visual recall tool. Without a cover or count, users must remember what every name contains; management receives more emphasis than opening content.

## Requirements

- Add a small cover mosaic and item count; make the whole card a clearly clickable browse target.
- Move Rename/Delete into a labelled overflow menu while keeping delete confirmation and the promise that originals are retained.
- Distinguish No saved items yet from Items hidden by filters, with Browse library or Show hidden items as appropriate.
- Keep saved ordering stable and show short confirmation when a picker adds/removes an item.

## Acceptance Criteria

- [ ] A user can distinguish populated/empty/filtered collections at a glance and reach the next useful action directly.
- [ ] Add red→green coverage for behavior changes; verify desktop and mobile UI.

## Implementation plan

- Preserve existing collection order (`id DESC`) and saved-item order (`membership id DESC`); repeated saves remain idempotent and do not reorder.
- Add opt-in `GET /api/collections?summary=true&include_nsfw=...` for index cards. Keep picker/plain list requests lightweight. Summaries add `item_count` (all saved), `visible_count` (filter-aware), and at most four newest visible `covers` with media identity and optional thumbnail path.
- Add unfiltered `item_count` to detail responses while retaining visible `total` for pagination. No schema change or persisted counters.
- Select/count covers using bounded set-based SQL, not full media hydration per card. Filter before selecting covers, including whole-video NSFW predicates across every sampled frame; never return hidden IDs or paths. Use image/frame thumbnail derivatives, with neutral missing-preview tiles.
- Add card browse targets and labelled management disclosure; preserve delete confirmation. Show distinct true-empty and filtered-empty states with Browse library / Show hidden items actions.
- Clear/gate stale cover and collection-detail responses on NSFW preference changes; reject late responses. Keep and test the existing picker success-only live notice rather than adding a new toast system.
- Split into backend contract/tests and frontend integration/tests commits. Cover filters/privacy/ordering/API compatibility and desktop/mobile keyboard behavior. Leave separately tracked video hydration optimization alone.

## Notes

- [HTML report, UX-07](../../docs/reviews/2026-09-22-atelier-ux-audit.html#UX-07) contains embedded screenshot evidence.

## Backend implementation

- Added opt-in filter-aware summaries and detail saved counts without migrations or changes to ordinary/picker list shapes. One set-based summary query limits cover output to four visible members and avoids full media hydration; work still scales with memberships.
- TDD: summary/detail field tests failed before implementation, then passed. Regression fixtures cover count meanings, hidden images/video tags/non-preview NSFW frames, image/video identity collisions, frameless videos, derivative paths, ordering and API compatibility. Targeted tests/race and the combined full repository suite passed.
- Independent backend review found no blockers. Representative query-plan/scale measurement and a shared-video multi-collection fixture are tracked separately; do not imply a measured performance regression.
