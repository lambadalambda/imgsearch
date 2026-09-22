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

- [ ] The exit action's text matches its destination in every launch context; loading, empty and playback errors are distinguishable.
- [ ] Add red→green coverage for behavior changes; verify desktop and mobile UI.

## Notes

- Audit only; no fix implemented.
- [HTML report, UX-10](../../docs/reviews/2026-09-22-atelier-ux-audit.html#UX-10) contains embedded screenshot evidence.
- Related launcher scope: [issue 062](062-surface-rail-feed-empty-unplayable-states.md).
