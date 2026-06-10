# Replace window.confirm delete confirmation with a styled dialog

## Summary

Deleting a pin uses the browser-native `window.confirm`, which clashes with the otherwise fully custom-styled overlays and cannot show the media being deleted.

## Requirements

- A small in-app confirmation dialog (consistent with Atelier styling) replaces `window.confirm`, showing the filename and a destructive-styled confirm button.
- Keyboard: Escape cancels, Enter confirms; focus moves into the dialog and returns on close (align with issue 086).
- Both delete entry points (card menu and, once issue 084 lands, the lightbox) use it.

## Acceptance Criteria

- Delete flow no longer calls `window.confirm`; smoke test drives the new dialog instead of `page.on("dialog")` auto-accept.
- Cancel leaves the pin untouched; confirm removes it (existing optimistic behavior).

## Notes

- `frontend/src/components/Pin.svelte:79-92`.
- Found during the 2026-06-10 UI/UX review.
