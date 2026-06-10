# Add seeking and time display to the Feed player

## Summary

The Feed progress bar is `aria-hidden` and non-interactive, and there is no elapsed/total time anywhere. Users cannot scrub within a video (native controls exist only in the lightbox), and taps on the bar do nothing.

## Requirements

- The progress bar becomes an interactive scrubber: click/drag (and touch) seeks the active video.
- Show elapsed / total time near the bar.
- Keyboard: Left/Right arrows seek ±5s while the feed is open (must not conflict with the existing Up/Down navigation).
- Scrubber gestures must not trigger the vertical swipe navigation.

## Acceptance Criteria

- Clicking at ~50% of the bar moves `currentTime` to ~50% of duration.
- Elapsed/total time renders and updates.
- Swiping vertically still navigates; dragging the scrubber does not.

## Notes

- `frontend/src/components/Feed.svelte:732-746` (progress bar), touch handling at lines 550-610.
- The existing `data-feed-chrome` guard already excludes chrome from swipe handling — the scrubber should sit inside it.
- Found during the 2026-06-10 UI/UX review.
