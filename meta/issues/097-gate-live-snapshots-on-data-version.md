# Gate live WebSocket snapshots on data changes and share them across clients

## Summary

`internal/live/http.go:139-178` recomputes `images.List`, `videos.List`, and `stats.Collect` (about a dozen queries, several of which full-scan `index_jobs` and `video_frames`) every 2s for every connected client, unconditionally. All of it runs on the single DB connection, serialized against the worker, so N open tabs cost N times the work.

## Requirements

- Skip recomputation when `PRAGMA data_version` (or an in-process change counter bumped by writers) has not changed since the last snapshot.
- Compute one snapshot per interval and broadcast it to all connected clients instead of one per client.

## Acceptance Criteria

- A test with two connected clients and no DB writes observes at most one snapshot query burst per interval.
- Clients still receive an update within one interval after a write.

## Notes

- Related: issue 099 (indexes) reduces the per-snapshot cost; this issue removes the multiplier.
- Found during the 2026-09-21 review.
