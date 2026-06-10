# Close the pin overflow menu on outside click

## Summary

The per-card "…" menu is a `<details>` element. It closes via Escape (only while focus is inside the card) or by re-clicking the summary — clicking anywhere else on the page leaves it open, and menus on several cards can be open simultaneously.

## Requirements

- Clicking/tapping outside an open pin menu closes it.
- Opening one pin's menu closes any other open pin menu.
- Escape continues to close the menu.

## Acceptance Criteria

- Open a pin menu, click on the page background → menu closes.
- Open pin A's menu, then pin B's → A's menu is closed.
- Smoke test covers the outside-click close.

## Notes

- `frontend/src/components/Pin.svelte:231-279`.
- A document-level pointerdown listener active only while open (or a shared "open menu" store) keeps this cheap.
- Found during the 2026-06-10 UI/UX review.
