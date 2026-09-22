# Add Atelier saved-collection workflows

## Summary

Frontend subissue of [saved media collections](save-media-to-named-collections.md), using the [collection API](persist-named-collections-and-membership.md).

## Requirements

- Collection navigation, create/rename/delete controls, and mixed-media collection browsing.
- Save/remove image/video membership through existing card/lightbox media actions; support discovery in the video feed too if its controls have an appropriate integration point.
- Clearly separate removing from collection from deleting media.
- Loading/empty/error states, focus/keyboard behavior, and safe stale-request handling.

## Acceptance Criteria

- Regression coverage proves creating a list, saving both media kinds, browsing after reload, and removing membership without deleting originals.
- Frontend checks, unit/browser tests, and independent correctness/maintainability review pass.

## Verification Notes

- Initial UI iteration passed 41 unit tests, zero-error/warning Svelte checks, production build, and the full Go/script/browser suite.
- Independent review findings were fixed with red→green regressions: management-list invalidation after Feed create, collection pagination after original-media deletion, and an in-flight Feed advance completing under the picker. Browser mock membership ordering/NSFW filtering now matches the API.
- Real local API/browser checks passed image/video save, mixed collection browsing, membership removal and reload, rename, and deletion with original records/files preserved. Desktop and 390px mobile layouts checked. Collection breadcrumb labels were corrected; Feed Save remains reachable over exhausted/error overlays.
- Final independent review approved correctness/architecture. All 44 frontend units, zero-error/warning Svelte checks, production build, `git diff --check`, and `mise run test:full` pass. Local browser follow-up verified creating/saving from exhausted Feed updates the already-mounted management list and saves the video ID.
- No production deployment performed.
