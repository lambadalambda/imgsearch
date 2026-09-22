# Persist named collections and media membership

## Summary

Backend/API subissue of [saved media collections](save-media-to-named-collections.md).

## Requirements

- Add forward-only SQLite migration for named collections and FK-backed image/video memberships.
- Authenticated API for create/list/rename/delete, idempotent add/remove, mixed contents, and media membership lookup.
- Validate names and IDs, bound pagination, reuse existing media response/NSFW conventions, and avoid duplicating media storage logic.
- Deleting collections only removes memberships; deleting media cascades its memberships.

## Acceptance Criteria

- TDD tests cover CRUD, invalid input, image/video membership, duplicate adds, missing resources, pagination, and deletion cleanup.
- Existing Go tests pass and an independent review approves correctness/maintainability.

## Notes

- Collections are shared within the authenticated instance, like its existing media library.

## Outcome

- Migration 16 persists collections and FK-backed mixed memberships. Added collection CRUD, membership lookup/mutations, and paginated hydrated items under existing API authentication.
- TDD: initial endpoint tests failed for missing implementation; runtime integration failed with 405 before route wiring. Focused tests then passed, including schema upgrade/reopen, NSFW-before-pagination, video-vs-frame identity, and actual media deletion cleanup/file preservation.
- `gofmt`, `git diff --check`, and `mise run test:full` passed. Independent review approved the backend; the non-blocking [video hydration scaling concern](bound-collection-video-hydration-work.md) is tracked separately.
