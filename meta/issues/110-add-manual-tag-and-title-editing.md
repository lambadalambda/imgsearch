# Add manual tag and title editing

## Summary

The only per-item mutations are delete, re-annotate, and toggle-NSFW (`internal/images/http.go:187-265`, `internal/videos/http.go`). Users cannot correct a wrong annotator tag, add their own, or rename a derived title. The Rail still shows "Tags (soon)" as a disabled button.

## Requirements

- `PATCH /api/images/{id}` and `PATCH /api/videos/{id}` accepting `title`, `tags` (full replacement), and keeping annotator-generated values distinguishable from user edits so a later re-annotate does not clobber manual tags (for example a `user_tags_json` column merged into the served `tags`).
- Also add `GET /api/images/{id}` and `GET /api/videos/{id}` single-item endpoints, which are needed for lightbox deep links.
- Lightbox gains an edit mode for title and a tag chip editor with add/remove.
- Tag search and the tag cloud include user tags.

## Acceptance Criteria

- Editing a tag in the lightbox persists across reload and appears in tag search.
- Re-annotating an item preserves user-added tags.
- Handler tests cover PATCH validation and the merge behaviour.

## Notes

- Found during the 2026-09-21 review.
- Outcome (2026-09-21): `tags_json` stays the served list so search, tag cloud, and NSFW SQL are unchanged; `annotator_tags_json`, `user_tags_json`, `removed_tags_json`, and `user_title` (migration 13) let `internal/mediaops` re-merge on re-annotation. The Rail "Tags (soon)" explorer button is a separate feature and stays disabled.
- Review follow-up: the single-item re-annotate endpoint now serves the user title and user tags while the job is pending instead of clearing them; `annotator_title` lets a cleared manual title revert immediately. Known limit: NSFW flags toggled before migration 13 count as annotator tags, so a later annotation that omits `nsfw` drops them (same as before the migration).
