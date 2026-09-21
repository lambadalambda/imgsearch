# Add a near-duplicate finder

## Summary

Deduplication is exact sha256 only (`internal/upload/service.go:198`). Resized, re-encoded, or lightly cropped copies of the same image are indexed as separate items. The embeddings already in the vector index give a cheap similarity signal, and libvips can produce a small perceptual hash at ingest.

## Requirements

- Compute a perceptual hash (dHash or pHash, 64-bit) at upload into a new `images.phash` column with a backfill job.
- Add `GET /api/duplicates` returning groups of items whose hashes fall within a configurable Hamming distance, optionally confirmed by embedding cosine similarity.
- Atelier view listing the groups side by side with per-item delete and a "keep largest" shortcut.

## Acceptance Criteria

- Uploading a JPEG and a resized WEBP copy places both in one duplicate group.
- Deleting from the duplicates view removes the item from the library as usual.

## Notes

- Found during the 2026-09-21 review.
- Outcome (2026-09-21): 64-bit dHash (`internal/phash`, pure Go via image/jpeg, image/png, image/gif, and golang.org/x/image/webp) stored in `images.phash` (migration 14, backfilled at startup; AVIF and sampled video frames are marked unhashable). `GET /api/duplicates?distance=N` groups by an 8-band index that is exact up to distance 7. The optional embedding-cosine confirmation was not built; the Atelier Duplicates view offers per-item delete and "Keep largest".
