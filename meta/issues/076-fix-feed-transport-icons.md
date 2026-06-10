# Fix Feed transport control icons

## Summary

All four transport buttons in the Feed overlay reuse the `feed` play-triangle icon with ad-hoc rotations:

- The play/pause button always shows a play triangle — it never switches to a pause glyph, so it gives no feedback about playback state.
- "Previous" renders as an up-pointing triangle, but "Next" renders as a right-pointing one. In a vertically-swiped feed, next should point down.

## Requirements

- Add proper `pause`, `chevron-up`/`chevron-down` (or equivalent) icons to `Icon.svelte`.
- The play/pause button reflects current playback state (play glyph when paused, pause glyph when playing).
- Previous points up, next points down, matching the swipe direction.

## Acceptance Criteria

- Visual: feed controls show distinct, directionally correct glyphs.
- The play/pause button's glyph (or accessible label/state) flips when playback is toggled; covered by the smoke test feed section.

## Notes

- `frontend/src/components/Feed.svelte:748-803`, `frontend/src/components/Icon.svelte`.
- Playback state is already tracked implicitly via the active `<video>`; a small `$state` flag fed by the existing `onplay`/`onpause` handlers is enough.
- Found during the 2026-06-10 UI/UX review.
