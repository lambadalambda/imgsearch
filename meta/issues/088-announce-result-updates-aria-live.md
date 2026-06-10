# Announce result updates to assistive technology

## Summary

The "N results · in X ms" line and the loading/error states render visually but are not announced — there is no `aria-live` region, so screen-reader users get no feedback after submitting a search or changing filters.

## Requirements

- The results meta line (loading / count / error) is an `aria-live="polite"` region (`role="status"` for normal updates; errors use `role="alert"` or assertive politeness).
- Announcements fire on search submit, tag/similar navigation, and filter changes, without being chatty during pagination.

## Acceptance Criteria

- The results meta element carries working live-region semantics; verified in the smoke test via attribute assertions.

## Notes

- `frontend/src/App.svelte:316-329`.
- Found during the 2026-06-10 UI/UX review.
