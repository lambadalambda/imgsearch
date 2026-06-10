# Add focus management to modal overlays

## Summary

Lightbox, Upload, and Feed set `role="dialog" aria-modal="true"` but never move focus into the dialog, don't trap Tab, and don't restore focus on close. Keyboard and screen-reader users keep tabbing through the page behind the overlay.

## Requirements

- On open, focus moves into the dialog (close button or the dialog container).
- Tab/Shift+Tab cycle within the dialog while it is open.
- On close, focus returns to the element that opened the dialog.
- Applies to Lightbox, Upload, and Feed overlays.

## Acceptance Criteria

- With the lightbox open, repeated Tab never reaches elements behind the overlay.
- Closing via Escape returns focus to the originating pin button.
- Smoke test asserts focus location after open and after close for at least the lightbox.

## Notes

- `frontend/src/components/Lightbox.svelte`, `Upload.svelte`, `Feed.svelte`. A small shared Svelte action (`use:focusTrap`) keeps it DRY.
- Found during the 2026-06-10 UI/UX review.
