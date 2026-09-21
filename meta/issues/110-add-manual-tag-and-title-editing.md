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
