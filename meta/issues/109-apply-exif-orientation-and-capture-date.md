# Apply EXIF orientation and extract capture date

## Summary

No EXIF handling exists anywhere in `internal/`. Rotated phone photos are embedded, thumbnailed, and displayed sideways, which degrades both search quality and browsing. There is also no capture date, so the only sort orders are random and upload time.

## Requirements

- Read EXIF orientation during upload and auto-rotate through the existing libvips pipeline before hashing derivatives, embedding, and serving. Keep the original bytes untouched on disk.
- Extract `DateTimeOriginal` (fall back to file mtime, then upload time) into a new `captured_at` column on `images`, with a backfill job for existing rows.
- Expose `captured_at` in API responses and add a "Captured" sort order in the Atelier header.

## Acceptance Criteria

- A fixture JPEG with orientation 6 renders upright in the grid and lightbox and its embedding matches the upright variant within tolerance.
- `captured_at` is populated for new uploads and backfilled for existing ones.
- Sorting by captured date works in the UI.

## Notes

- Interacts with issue 079 (thumbnail derivatives): apply orientation there too.
- Found during the 2026-09-21 review; first feature to build.
