# Bound collection video hydration work

## Summary

Backend collection review identified a non-blocking scaling concern: a saved-items page hydrates each video via `videos.GetItem`, whose aggregate/window CTEs currently consider library-wide frames and jobs. Video-heavy pages can repeat this work up to 200 times.

## Requirements

- Profile a representative video-heavy collection before changing the query strategy.
- If warranted, restrict single-video hydration CTEs to the requested ID or add bounded batch hydration while retaining one canonical media response implementation.
- Preserve preview selection, annotation/job state, transcript, and NSFW behavior.

## Acceptance Criteria

- A representative benchmark demonstrates whether repeated hydration is a bottleneck.
- Any optimization has regression coverage for response parity and demonstrates bounded work without copying the library SQL into collections.

## Notes

- Found during independent saved-collections backend review; not a correctness blocker.
- Relevant paths: `internal/collections/store.go` item hydration and `internal/videos/http.go` `GetItem`/`queryVideos` CTEs.
