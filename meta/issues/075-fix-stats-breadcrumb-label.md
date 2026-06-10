# Fix Statistics breadcrumb showing "Similar"

## Summary

On the Statistics view the header breadcrumb reads "Library / Discover / Similar". The breadcrumb in `Header.svelte` handles `library`, `search`, and `tag` modes explicitly and falls through to a `{:else}` branch that renders "Similar" for every other mode, including `stats`.

## Requirements

- The stats view must show a breadcrumb that names the stats view (e.g. "Library / Statistics").
- The similar view keeps its current "Similar" crumb.

## Acceptance Criteria

- Navigating to `?view=stats` renders a breadcrumb without "Similar".
- UI smoke test asserts the stats breadcrumb label.

## Notes

- `frontend/src/components/Header.svelte:34-38`.
- Found during the 2026-06-10 UI/UX review.
