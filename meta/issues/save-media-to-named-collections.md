# Save images and videos to named collections

## Summary

Let users collect media they discover into persistent named lists, then browse those lists later in Atelier.

## Requirements

- Create, rename, and delete named collections in the existing single-library/auth model (no new user/sharing system).
- Save images and videos to one or more collections through existing media actions.
- Browse mixed-media collection contents and remove memberships without deleting underlying files/media.
- Persist membership in SQLite, prevent duplicates, and clean up references when media or collections are deleted.
- Keep the existing library/search/feed/favorite behaviors intact.

## Acceptance Criteria

- Backend regression tests cover collection CRUD, validation, mixed membership, duplicate adds, missing media, and safe deletion/cleanup.
- Atelier supports collection management and saving/removing both media kinds, with loading/empty/error states.
- Regression tests exercise collection UI integration, including persistence across reload.
- Formatting, frontend checks, and full tests run; independent correctness/maintainability review completed.

## Notes

- Collections reference existing media; they do not copy files, import external URLs, or schedule synchronization.
- Scope is the Atelier frontend, not a parallel implementation in the legacy shell.
