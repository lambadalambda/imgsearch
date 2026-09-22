# Save images and videos to named collections

## Summary

Let users collect media they discover into persistent named lists, then browse those lists later in Atelier.

## Requirements

- Create, rename, and delete named collections in the existing single-library/auth model (no new user/sharing system).
- Save images and videos to one or more collections through existing media actions.
- Browse mixed-media collection contents and remove memberships without deleting underlying files/media.
- Persist membership in SQLite, prevent duplicates, and clean up references when media or collections are deleted.
- Keep the existing library/search/feed behaviors intact.

## Acceptance Criteria

- Backend regression tests cover collection CRUD, validation, mixed membership, duplicate adds, missing media, and safe deletion/cleanup.
- Atelier supports collection management and saving/removing both media kinds, with loading/empty/error states.
- Regression tests exercise collection UI integration, including persistence across reload.
- Formatting, frontend checks, and full tests run; independent correctness/maintainability review completed.

## Notes

- Collections reference existing media; they do not copy files, import external URLs, or schedule synchronization.
- Scope is the Atelier frontend, not a parallel implementation in the legacy shell.

## Subissues

- [Persistence and API](persist-named-collections-and-membership.md)
- [Atelier workflows](add-atelier-saved-collection-workflows.md)

## Outcome

- Both subissues completed and independently reviewed. Backend commit `c180bff` provides migration/API; Atelier adds collection navigation/management and a shared picker for cards, lightbox, and the current Feed video.
- Verified persisted mixed membership, idempotent saves, safe deletion, NSFW filtering, pagination, error/retry and keyboard/focus behavior. TDD regression coverage includes review/QA findings.
- Final frontend check/build and full Go/frontend/script/browser suite pass; focused backend race checks pass. Manual real-API browser checks passed on desktop and mobile. Documentation: [collections](../../docs/collections.md).
- Non-blocking [video hydration scaling](bound-collection-video-hydration-work.md) remains a separate follow-up. No deployment or production data changes made.
