# 066: Optimize Tag Cloud Handling

## Priority

P2

## Status

Open.

## Summary

The tag-cloud endpoint computes tag counts by scanning JSON tag arrays across the media library. On larger libraries this can take seconds, and if it starts during initial UI bootstrap it can block the first `/api/images` request behind SQLite's single connection.

## Context

- Atelier bootstrap requests `/api/search/tag-cloud?limit=16` for the QuickRow tag chips.
- The backend tag-cloud query uses `json_each` over image/video tag JSON.
- The runtime uses a single SQLite connection, so a slow tag-cloud read can block otherwise fast page-list reads.
- Deferring the frontend request fixes the initial render symptom, but tag counting still needs a better long-term data model.

## Risks

- Initial page load can show skeletons until the tag-cloud scan finishes.
- Tag-chip counts get more expensive as the library grows.
- NSFW-aware tag counts and tag search behavior can drift if tag handling remains ad hoc across JSON scans.

## Acceptance Criteria

- [ ] Add timing/regression coverage for initial page load not being blocked by tag-cloud work.
- [ ] Design a materialized or normalized tag-count representation for images/videos.
- [ ] Keep tag counts scoped consistently for standalone images, video media units, and NSFW filtering.
- [ ] Update tag cloud, tag search, and any affected stats/jobs paths to use the chosen tag model.
- [ ] Document the tag handling model and invalidation/backfill behavior.

## Related Files

- `frontend/src/App.svelte`
- `internal/search/http.go`
- `internal/db/migrations.go`
- `internal/worker/queue.go`
- `scripts/ui_smoke_test_atelier.mjs`
