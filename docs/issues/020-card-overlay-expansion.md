# 020: Add Hover/Focus Overlay Expansion for Clipped Card Content

## Status: Completed

## Goal

Allow long card content to be explored without destabilizing the grid layout by expanding clipped content in an overlay that sits on top of neighboring cards rather than reflowing the page.

## Problem

The current inline expand/collapse pattern is functional, but it pushes card height around and breaks grid rhythm. That makes the library feel more like a document list than a polished media browser.

## Desired Direction

- Cards keep a stable resting height in the grid.
- On hover or focus-within, the active card can expand visually over neighbors.
- Expanded content should:
  - reveal full title/prose/tags as needed
  - keep the grid stable
  - feel like a floating detail layer on top of the base card

## Scope

- Introduce a card inner/overlay structure if needed
- Replace the current inline "Show more" expansion pattern where appropriate
- Support both hover and keyboard focus
- Keep reduced-motion behavior tasteful and accessible

## Scope Landed

- Added a dedicated `card-detail-overlay` layer per card that sits absolutely above the base grid card and is hidden in rest state.
- Kept card resting dimensions fixed while expanded content overlays neighbors instead of reflowing rows.
- Added hover + `:focus-within` activation so pointer and keyboard interactions reveal the same expanded content affordance.
- Expanded overlay content includes full title/path/support text, unclamped video meta text, and wrapped tag rows.
- Tuned motion to subtle opacity/translate transitions and disabled those transitions under `prefers-reduced-motion`.

## Acceptance Criteria

- [x] Cards keep a stable resting size in the grid
- [x] Hover/focus expansion does not push neighboring cards around
- [x] Expanded content remains readable and layered correctly above the grid
- [x] Keyboard users can access the same expansion affordance via focus

## Notes

- This issue should build on the card-height standardization work, not duplicate it.
- The animation should be subtle and product-like, not flashy.
