# Fix Feed and lightbox keyboard and advance conflicts

## Summary

Three related interaction bugs found in the same review:

- Feed "Next" at the tail silently no-ops while a lookahead fetch is in flight: `Feed.svelte:271-280` awaits `ensureLookahead()`, but line 393 returns immediately without awaiting the existing `lookaheadPromise`. Feedback is recorded, nothing advances, and the user must tap again.
- Space closes the Feed right after opening: the focus trap focuses the first focusable (`focus.ts:24`), which is the close button, and `Feed.svelte:545-547` ignores Space when the target is a button, so native activation closes the overlay.
- Lightbox arrow keys hijack native video seeking: `Lightbox.svelte:117-123` calls `preventDefault` on ArrowLeft/Right at the window level, so a keyboard user focused on `<video controls>` navigates to the neighbouring pin instead of seeking.

## Requirements

- `ensureLookahead` returns the in-flight promise when one exists.
- Feed initial focus goes to the overlay root or the play/pause button, not the close button.
- Lightbox ignores arrow keys when `event.target` is inside the video element.

## Acceptance Criteria

- Smoke tests cover each of the three scenarios.

## Notes

- Found during the 2026-09-21 review.
