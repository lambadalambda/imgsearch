# 020: Add Hover/Focus Overlay Expansion for Clipped Card Content

## Status: Open

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

## Acceptance Criteria

- [ ] Cards keep a stable resting size in the grid
- [ ] Hover/focus expansion does not push neighboring cards around
- [ ] Expanded content remains readable and layered correctly above the grid
- [ ] Keyboard users can access the same expansion affordance via focus

## Notes

- This issue should build on the card-height standardization work, not duplicate it.
- The animation should be subtle and product-like, not flashy.
