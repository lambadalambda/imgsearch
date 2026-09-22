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

- [x] A user can distinguish populated/empty/filtered collections at a glance and reach the next useful action directly.
- [x] Add red→green coverage for behavior changes; verify desktop and mobile UI.

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

## Frontend completion

- Added browse cards with safe lazy thumbnail mosaics, saved/hidden counts and neutral unavailable previews; sibling labelled management disclosures preserve confirmation and keyboard focus. Shared pure presentation helpers distinguish empty/filtered/partially filtered states without treating errors as empty.
- True-empty details offer Browse library; fully filtered details explain and offer the global Show hidden items preference. Existing picker success-only live notices and stable saved order remain covered.
- Scope guards clear stale index/detail/lightbox content immediately on visibility changes and reject late responses, including failed replacements and requests that ignore abort.
- TDD: missing summary API/helper tests and summary-request smoke failed first; implementation passed. Independent review found inaccessible summary text beneath browse aria-labels; a new Chromium accessible-description assertion failed, then passed after adding unique aria-describedby targets. New race-test waits are bounded and cleaned up. Follow-up review approved with no blockers.
- Verified fresh frontend build/asset gate, 49 unit tests, full Go/script/browser suite, frontend check (zero errors/warnings), focused collection race tests, formatting and whitespace. Browser smoke covers counts/states/CTAs, menu keyboard/focus/cancellation, picker feedback, saved order, and delayed/failed filtering.
- Desktop1280x900 and mobile390px screenshots inspected; overflow assertions passed. Evidence is local mock-server/synthetic-media QA, not real-photo aesthetics or a production data test. Open-lightbox preference changes have a source gate but no dedicated browser regression in this pass.
- Archived after implementation/review. No deployment or push; production still runs token-settings release `cc0aab4`.
