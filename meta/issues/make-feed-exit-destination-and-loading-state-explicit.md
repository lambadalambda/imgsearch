# “Back to library” should go to the Library

## Priority

P2 — Reproduced workflow issue.

## Summary

The end-of-feed action closes the overlay, returning to the underlying page—even when that page is Settings.

## Evidence

Twice in the sandbox: launch Feed from Settings, advance to the last item, click Back to library → the URL remains ?view=settings. Separately, captured mobile Feed shows a truncated title, 0:00 / 0:00 and Pause while metadata is unavailable. Playback did not initialize in this automation environment; no production playback defect is inferred.

Reproduction / inspection steps:

1. Open Settings, then start Feed from the rail.
2. Advance to the terminal item and click Back to library.
3. Observe that Settings is still the destination.

## Risks

The exit label promises a destination it does not deliver. Loading and playback state should also be understandable without interpreting a zero-duration transport.

## Requirements

- Either navigate to Library or rename the action Close feed / Back to Settings according to the actual destination.
- Related state improvement: show explicit Loading video / Cannot play states; do not leave an apparently playable zero-duration transport indefinitely.
- Related layout improvement: keep the video title readable on phones; consider a compact Save control with an accessible label.
- Handle empty/unplayable launcher outcomes explicitly (existing issue 062); verify playback and gestures on real mobile browsers before changing media logic.

## Acceptance Criteria

- [x] The exit action's text matches its destination in every launch context; loading, empty and playback errors are distinguishable.
- [x] Add red→green coverage for behavior changes; verify desktop and mobile UI.

## Notes

- Integrated with the new navigation; independent source and integration reviews found no blockers. Mobile close restores More; desktop close restores Feed.
- All Feed close actions now say Close feed, preserving the underlying Settings, Library or search context rather than promising navigation. Mobile title wraps; the compact Save button retains the accessible name Save to collection. Initial focus moves to the title because transport is unavailable before metadata.
- Metadata/loading and decoding failures have distinct Loading video… / Cannot play this video feedback. Unknown/failed duration hides seek/time and disables play; metadata events and buffered-slot promotion restore usable transport. Terminal overlays hide underlying transport from keyboard focus while keeping Save, Close and terminal actions reachable. Existing queue, feedback, autoplay, seek mechanics and gestures are otherwise unchanged.
- Narrow issue 062 overlap: Rail now gives dismissible No videos available / Could not open feed feedback instead of silently returning. Its busy Feed trigger uses guarded `aria-disabled` rather than losing focus through native disabling, so overlay close restores the visible launcher. No App edits.
- TDD evidence: `scripts/atelier_feed_test.mjs` first failed waiting for Loading video…. A further red→green terminal keyboard test caught inert controls trapping focus behind the terminal overlay. Green coverage includes metadata pending/success/failure, next/previous state reset, fetch failure/retry vs empty terminal, Settings-launched close/focus restoration, Library/search close destinations, empty/failed launcher notices, readable mobile title, and keyboard access to Save/Close/terminal actions.
- Verification: frontend unit suite (49 tests), Svelte check (0 errors/warnings), validated production build, full Atelier browser smoke and asset/build regressions (10 tests). Focused rerun: `UX_ONLY=feed npm test` after building. The smoke's old PNG-as-video transport fixtures now supply explicit synthetic metadata instead of relying on an active zero-duration transport.
- Screenshot evidence (local artifacts, not committed): `/tmp/atelier-ux06-ux10/feed-{loading,ready,cannot-play,failed,empty}-mobile.png`, `feed-launch-{empty,failed}-mobile.png`, `feed-ready-desktop.png`, `feed-local-video-{mobile,desktop}.png`. Controlled state fixtures use a local cat poster. The optional `UX_ONLY=feed UX_REAL_MEDIA=1 npm test` decoder check played the repository's local MP4 fixture in this Chromium run; it is opt-in because valid media can remain uninitialized in automation.
- Limits: physical iOS/Android browsers, OS keyboard and real touch gestures are not verified. No production playback diagnosis or broad gesture/launcher rewrite; no backend or llama.cpp work. The local MP4 success is evidence for the fixture/browser only.
- [HTML report, UX-10](../../docs/reviews/2026-09-22-atelier-ux-audit.html#UX-10) contains embedded screenshot evidence.
- Related launcher scope: [issue 062](062-surface-rail-feed-empty-unplayable-states.md).
- Parent integration validation: `mise run test:full`, `mise run check:frontend` (0 errors/warnings) and `git diff --check` passed, including all 61 frontend unit tests and mobile More focus restoration.
