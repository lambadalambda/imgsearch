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
