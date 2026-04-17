# 021: Move Card Actions into Thumbnail Overlays and Simplify Rest State

## Status: Completed

## Goal

Reduce the default visual clutter in image/video/result cards by moving secondary actions out of the meta stack and into a lighter thumbnail-overlay affordance.

## Problem

Even after small polish passes, the rest-state cards still devote visible space to controls like:

- similar-image action
- delete action

These controls contribute to card-height variation and make the resting grid feel more tool-like than gallery-like.

## Desired Direction

- Keep cards visually calmer at rest
- Surface actions on hover/focus, ideally in the thumbnail corner
- Preserve keyboard accessibility and discoverability
- Keep destructive actions clearly separated from search/navigation actions

## Scope

- Reposition Similar/Delete actions into a thumbnail-level overlay or compact action region
- Reduce footer/button dominance in the default card state
- Ensure the action system still works for:
  - image cards
  - video cards
  - search result cards where appropriate

## Scope Landed

- Moved `Find similar` and delete actions from the card meta stack into a `thumb-actions` overlay anchored to the thumbnail region.
- Added hover + keyboard-focus activation so actions remain hidden at rest and become available on interaction.
- Preserved media-type behavior:
  - images: similar + delete
  - videos: similar + delete
  - search results: similar/anchor action where available, no delete button
- Kept delete confirmation and delete API flow unchanged.
- Reduced rest-state control weight in card metadata to keep the grid calmer and more gallery-like.

## Acceptance Criteria

- [x] Similar/Delete actions no longer dominate the default card layout
- [x] Actions remain discoverable and accessible with mouse and keyboard
- [x] Hover/focus action presentation feels lighter and more product-like
- [x] Delete confirmation behavior remains unchanged

## Notes

- This issue should follow the card-height/content-density work so the card surface is already simplified first.
