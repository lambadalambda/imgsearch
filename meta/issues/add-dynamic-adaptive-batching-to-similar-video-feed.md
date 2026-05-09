# Add dynamic adaptive batching to the similar-video feed

## Summary

Make the similar-video feed feel more dynamic by starting with a small seed-anchored queue and lazily appending small batches of new candidates as the user swipes. New batches should use the current session's inferred preferences while keeping all feedback private and session-local.

## Requirements

- Start feed sessions with a small initial queue: the selected seed video plus about four similar candidates.
- Maintain a lookahead buffer so new candidates are requested before the user reaches the end of the queue.
- Fetch additional videos in small batches, preferably three at a time, when the remaining queued items ahead of the current item fall below the buffer threshold.
- Keep all adaptation session-local in browser memory; do not persist feed feedback or create user profiles.
- Send seen and already-queued video IDs to the backend so batches do not repeat videos within the same session.
- Use current session preference signals when fetching new batches: high completion / ended videos should nudge future results toward matching tags, while quick skips should weakly demote matching tags.
- Keep visual similarity as the primary candidate-generation signal; tag preferences should nudge ranking within the visually similar pool, not replace similarity.
- Clamp and decay session tag preferences to avoid overfitting to one early watch-through or skip.
- Preserve playback stability: do not reorder the current item or the immediately preloaded next item while the user is watching.
- Handle stale or overlapping fetch responses without duplicating candidates or corrupting the queue.
- Preserve the existing end-of-feed behavior when no more candidates can be found.

## Acceptance Criteria

- Opening a feed queues the seed plus no more than four initial candidates.
- Advancing near the end of the queued buffer triggers a background request for another small batch.
- Newly fetched videos append to the tail and exclude all seen or already queued videos.
- Positive feedback increases preferred tag influence for later backend batches.
- Quick-skip feedback weakly demotes avoided tags for later backend batches.
- Tag scores are clamped and decay over time so early feedback does not permanently dominate.
- Backend similar-video batches accept preference hints and rerank candidates without filtering out all visually similar results.
- Concurrent or stale batch fetches cannot append duplicate videos.
- UI/browser smoke coverage exercises lazy feed loading and queue appending.
- Go tests cover preference-aware similar-video reranking and seen/queued exclusion.

## Notes

- Advisor consensus: use candidate generation plus reranking, not a full recommender or persistent profile. Keep embedding similarity primary, then apply session tag hints and simple diversity/exploration guardrails.
- Google’s recommendation-system overview describes a common candidate-generation, scoring, and re-ranking pipeline; this maps well to the existing vector-search endpoint plus lightweight tag reranking.
- MMR-style diversity reranking is a useful future guardrail for reducing near-duplicate batches, but the MVP should keep the scoring simple and inspectable.
- Kuaishou’s CIKM 2023 short-video recommender paper models very short watch/skip behavior as implicit negative feedback. The feed should keep negative signals weak because a skip can mean “not now” rather than true dislike.
- Suggested MVP scoring: visual similarity first, then preferred-tag boost, avoided-tag penalty, tiny deterministic exploration/diversity nudge if needed.
- Suggested defaults: initial candidates `4`, batch size `3`, fetch threshold `2`, positive tag weight `+1`, soft-negative tag weight `-0.5`, tag score clamp `[-3, 5]`, decay `0.9` per fetch.

## References

- https://developers.google.com/machine-learning/recommendation/overview/types
- https://qdrant.tech/blog/mmr-diversity-aware-reranking/
- https://fi.ee.tsinghua.edu.cn/~gaochen/papers/CIKM2023.pdf
