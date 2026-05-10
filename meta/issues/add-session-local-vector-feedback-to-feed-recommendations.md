# Add session-local vector feedback to Feed recommendations

## Summary

Explore and implement a session-local Feed recommendation mode that adapts the query vector from watch and skip feedback instead of relying only on tag preference reranking.

Stored media embeddings must remain immutable. The adaptation should build a temporary session query from the seed video plus positive and soft-negative feedback, then use that query for future similar-video candidate searches.

## Requirements

- Keep all vector feedback session-local and wipe it when Feed closes.
- Do not mutate stored image, video-frame, transcript, or model embeddings.
- Build the adaptive query from existing local embeddings, for example seed vector plus recent positive vectors minus weak soft-negative vectors.
- Keep the current video and immediately preloaded next video stable when feedback changes.
- Preserve tag scoring as a bounded, interpretable nudge unless evidence shows it should be replaced.
- Keep visual similarity anchored so early noisy feedback cannot cause severe topic drift.
- Handle videos with multiple sampled frames deliberately, such as first indexed frame, matched frame, representative frame, or averaged frame vectors.
- Respect existing NSFW, seen-video, playable-video, and session privacy behavior.

## Acceptance Criteria

- Feed can request candidates from a session-adapted query vector without changing persisted embeddings.
- Positive watch-through feedback increases the rank of visually similar future candidates even when tags are absent or unhelpful.
- Quick-skip soft-negative feedback weakly reduces visually similar future candidates without over-filtering the feed.
- Query-vector adaptation is bounded, normalized, and tested against early-session drift.
- Backend tests cover vector-query candidate search, seen filtering, NSFW filtering, and the interaction between vector feedback and tag reranking.
- Frontend or smoke coverage verifies that feedback changes later batches while keeping the current and next item stable.

## Notes

- The current Feed endpoint uses `SearchByImageID` from the original seed frame and applies only bounded `prefer_tags` / `avoid_tags` reranking.
- The vector index already exposes `Search(ctx, modelID, query, limit)`, so the main backend work is retrieving/combining suitable feedback vectors and exposing a safe request shape.
- A Rocchio-style formula is a reasonable starting point: `normalize(seed + alpha * mean(positive) - beta * mean(soft_negative))`, with `beta` smaller than `alpha`.
- This should be a deliberate follow-up to tag-only adaptation, not a change to persistent user profiles.

## Implementation Progress

- Added stateless `positive_image_ids` and `soft_negative_image_ids` request parameters for similar-video Feed batches.
- Backend builds a bounded normalized session query from the seed frame plus validated feedback frame embeddings, then falls back to `SearchByImageID` when no valid feedback is available.
- Atelier and legacy Feed keep recent positive and soft-negative frame IDs in memory, clear them on close/start, and send them only with later batch requests.
- Added backend and smoke coverage for positive feedback, soft-negative feedback, overlap handling, seen/NSFW filtering, tag reranking, and frontend request plumbing.
