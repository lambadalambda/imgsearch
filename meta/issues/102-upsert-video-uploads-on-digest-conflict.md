# Upsert video uploads on sha256 conflict

## Summary

`internal/upload/service.go:275-282` selects by sha256 and then inserts without `ON CONFLICT` (lines 318-321). Two concurrent uploads of the same video both run ffmpeg, and the second fails with a UNIQUE error reported as a generic upload failure instead of `duplicate: true`. Images already use the conflict-safe path at line 254.

## Requirements

- Use the same `INSERT ... ON CONFLICT` pattern as images so the loser of the race gets the existing row and `duplicate: true`.
- Avoid running ffmpeg for the loser where possible (check again inside the transaction before sampling, or accept the wasted work and just fix the response).

## Acceptance Criteria

- A test uploading the same video concurrently returns one insert and one duplicate response, with no error.

## Notes

- Found during the 2026-09-21 review.
