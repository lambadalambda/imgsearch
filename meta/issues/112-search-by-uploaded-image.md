# Search by an uploaded or pasted image

## Summary

`/api/search/similar` only accepts an `image_id` that is already in the library (`internal/search/http.go:244-251`). There is no way to find matches for a picture the user has on disk or on the clipboard without importing it first.

## Requirements

- `POST /api/search/by-image` accepting a multipart image, embedding it in-process with the query embedder without persisting it, and returning the same result shape as `similar`.
- Search bar accepts a pasted image or a dropped file and switches to a "Similar to upload" mode with a small preview of the query image in the provenance line.
- Respect the same size limits and NSFW filtering as the rest of search.

## Acceptance Criteria

- Pasting an image into the search bar returns ranked results without creating a library item.
- Handler test covers a fake embedder returning deterministic top results.

## Notes

- Found during the 2026-09-21 review.
- Outcome (2026-09-21): the query image is spooled under `data/tmp/`, embedded with the configured image embedder, and deleted; the `byimage` view is not restored from the URL after a reload because the file only lives in memory.
