# 019: Standardize Library Card Height and Content Density

## Status: Completed

## Goal

Make the Images, Videos, and Search Results grids feel orderly and polished by giving cards a much more consistent resting size and reducing metadata density in the default view.

## Problem

Cards currently vary widely in height because they can stack:

- title
- path
- description
- transcript text
- tags
- video metadata
- status chips
- score text
- action buttons

This creates ragged rows, weak visual rhythm, and a cluttered engineering-tool feel.

## Desired Direction

- Give cards a fixed or narrowly bounded resting height.
- Clamp long content aggressively at rest:
  - title
  - path
  - supporting text
  - tags
- Reduce the number of visible metadata lines per card.
- Prefer one supporting text slot at rest rather than multiple stacked prose blocks.

## Scope

- Rework card layout in `internal/webui/static/app.js` and `styles.css`
- Tighten title/path/supporting-text/tag presentation
- Normalize thumbnail-to-meta proportions across Images/Videos/Results cards

## Scope Landed

- Reworked card rendering to use one compact supporting-text slot at rest instead of stacking description + transcript blocks.
- Removed inline card expansion controls from rest-state cards so grid rows keep a consistent silhouette.
- Added fixed resting media/meta proportions for cards in all grids (Images, Videos, Results).
- Clamped title/path/supporting text and video metadata at rest.
- Changed tag rendering to a single-row compact list with a `+N` overflow chip to prevent multi-row card growth.
- Kept existing search, delete, and media-open behaviors intact.

## Acceptance Criteria

- [x] Cards in grid views have a much more consistent resting height
- [x] Titles, paths, and prose blocks are clipped at rest instead of growing the grid unpredictably
- [x] Tag rows no longer create multi-row card-height explosions by default
- [x] The card layout still works for images, videos, and search results

## Notes

- This issue is about the resting card silhouette and density.
- Hover/focus expansion behavior for clipped content is tracked separately.
