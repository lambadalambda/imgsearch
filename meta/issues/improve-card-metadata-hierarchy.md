# Improve card metadata hierarchy

## Summary

Image and video cards currently make filenames and storage paths more prominent than semantic descriptions and tags. This makes the gallery read like a file listing instead of a content-browsing interface.

## Requirements

- Promote available image/video descriptions to the primary card title.
- Demote filenames to secondary metadata or hover detail, and avoid showing storage paths in the resting card state.
- Keep tags readable in compact cards instead of truncating most labels to tiny fragments.
- Make common index states user-facing and less visually dominant.
- Preserve hover/focus detail access to filenames, paths, status, tags, and supporting text.

## Acceptance Criteria

- Cards with descriptions show the description as the visible primary title.
- Cards without descriptions still fall back to the original filename.
- Resting cards do not show storage paths by default.
- Compact tags remain legible on desktop and mobile.
- Existing keyboard, hover, lightbox, and search-similar flows continue to work.

## Notes

- Review feedback from local desktop/mobile screenshots called this the highest-impact shippable improvement.
