# Add a re-annotate-all endpoint

## Summary

Re-annotation exists per item (`internal/mediaops/reannotate.go`). After switching annotation backends users need a way to refresh the whole library on demand, since switches deliberately keep existing annotations.

## Requirements

- `POST /api/jobs/reannotate-all` that sets `reannotate_requested` and resets annotation jobs to `pending` for every standalone image and every video, in one transaction, reusing `RequestReannotationJob`.
- Optional `media` filter (`images`, `videos`, `all`, default `all`).
- Returns the count of queued jobs; idempotent when jobs are already pending.
- Skips jobs currently `leased` (same behaviour as per-item re-annotate) and reports them separately.

## Acceptance Criteria

- Handler test: library with images and videos, call the endpoint, all annotation jobs are `pending` with `reannotate_requested = 1`, response counts match.
- Statistics pane shows the queue growing after the call.

## Notes

- Subissue of [113](113-annotation-backend-settings-page.md). Independent of 116 and 117.
- Consider a confirmation-worthy warning in the UI: on paid remote APIs this costs money.
