# Add lookup indexes for `index_jobs`

## Summary

The partial unique indexes on `index_jobs` cannot serve queries that do not repeat their `WHERE` clause. EXPLAIN confirms `DELETE FROM index_jobs WHERE image_id = ?` (`internal/images/http.go:398`, `internal/videos/http.go:502`) full-scans, the `videos.List` CTEs full-scan, `images.List` builds an automatic index on every call, and the claim queries (`internal/worker/queue.go:119-130, 200-210`) sort all runnable rows in a temp b-tree.

## Requirements

- Add a migration with `index_jobs(image_id, model_id, kind)`, `index_jobs(video_id, model_id, kind)`, and `index_jobs(state, kind, run_after, created_at)`.
- Verify with EXPLAIN QUERY PLAN that the delete, list, and claim paths use them.

## Acceptance Criteria

- Migration test applies cleanly on an existing DB.
- EXPLAIN for the four query shapes above no longer shows `SCAN index_jobs` or `AUTOMATIC` indexes.

## Notes

- Found during the 2026-09-21 review.
