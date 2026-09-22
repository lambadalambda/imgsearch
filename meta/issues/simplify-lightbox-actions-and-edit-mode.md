# Separate viewing actions from editing actions

## Priority

P2 — Design recommendation.

## Summary

Save, Save to collection, Find similar, re-annotation and deletion compete in the same small action area—even during editing.

## Evidence

The desktop editor shows Save to collection beside Save, then Cancel / Find similar / Open original / Flag NSFW / Re-annotate / Delete. On mobile, a long hash filename and description push these actions below the visible sheet. Actual title/tag editing and saving succeeded in the sandbox.

Reproduction / inspection steps:

1. Open an image and choose Edit.
2. Compare the action area with the phone lightbox's initial viewport.

## Risks

Two different “save” concepts compete, and unrelated actions distract from finishing the edit. A filename that is mostly a hash has little reading value but consumes prime mobile space.

## Requirements

- During editing, emphasize Save changes and Cancel; move unrelated actions out of the edit flow.
- In viewing mode, prioritize Save to collection, Similar, and More; keep destructive/reprocessing actions in the latter.
- Make a compact action row sticky within the mobile sheet and collapse filename/technical fields under File details.
- Retain the current title/summary/description hierarchy and previous/next navigation—they are useful.

## Acceptance Criteria

- [x] Save changes is unambiguous and reachable; core viewing actions remain easy to find without reading the entire description.
- [x] Add red→green coverage for behavior changes; verify desktop and mobile UI.

## Notes

- Implemented in the isolated UX06/UX10 worktree; awaiting parent review/application. Index/archive/changelog intentionally untouched.
- Viewing: Save to collection, Similar, More and Edit remain visible in a compact, fixed-within-sheet action row. More contains Open original, NSFW, Re-annotate and Delete. Editing replaces those actions with Save changes / Cancel, including disabled Saving… feedback; failed saves retain the draft.
- Filename is collapsed in keyboard-accessible File details. Title, summary, description, tags and previous/next navigation remain. Editing focuses the title; cancel/success restores Edit; Escape dismisses More before the lightbox. Match badges reuse the parent-owned `relativeMatchLabel` helper (`Match NN/100`, with a relative-similarity explanation), not percentages.
- TDD evidence: `scripts/atelier_lightbox_test.mjs` first failed because Delete was visible without More. The green test covers long hash filenames and long descriptions, collapsed details, menu keyboard behavior, failed/pending/successful edits, save/cancel reachability and focus, previous/next, close focus restoration and 320/390px mobile layouts. Existing smoke checks now explicitly open More for advanced actions.
- Verification: frontend unit suite (49 tests), Svelte check (0 errors/warnings), validated production asset build, full Atelier browser smoke, and asset/build regression suite (10 tests). Focused rerun: `UX_ONLY=lightbox npm test` after building.
- Screenshot evidence (local artifacts, not committed): `/tmp/atelier-ux06-ux10/lightbox-desktop.png`, `lightbox-view-{320,390}.png`, `lightbox-edit-{320,390}.png`, `lightbox-edit-failed-mobile.png`, `lightbox-more-mobile.png`, and `lightbox-edit-short-viewport.png`. Uses the repository's `fixtures/images/cat_1.jpg` and isolated API fixtures.
- Limits: Chromium desktop/resized mobile and reduced-height viewport, not a physical phone/OS soft keyboard. No backend, App, collection behavior, shared match-helper implementation or media-navigation changes.
- [HTML report, UX-06](../../docs/reviews/2026-09-22-atelier-ux-audit.html#UX-06) contains embedded screenshot evidence.

## Completion

Independent review approved; parent integrated with navigation/Settings/duplicate changes and full repository/browser/frontend checks passed. Lightbox now uses the shared relative Match score helper. Desktop/mobile320/390px and reduced-height fixture screenshots inspected; no real-device keyboard certification. Archived. Existing cross-item action feedback is separately tracked.
