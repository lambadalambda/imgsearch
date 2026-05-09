# Add mobile adaptive similar-video feed mode

## Summary

Add a mobile-first, TikTok-style continuous video mode that can be started from any video. The mode should build a queue of similar local videos, let the user swipe through them, and use session-local watch behavior to adapt future queue ordering.

This should be implemented as an adaptive local video browser, not a social feed clone. The feature should reuse imgsearch's existing local embeddings/search data and keep feedback private by default.

## Requirements

- Provide an entry point from video cards, video search results, and/or the existing video player to start the feed from a specific seed video.
- Render a fullscreen mobile feed with one video active at a time, muted `playsinline` autoplay by default, a clear exit control, and a visible queue-position indicator.
- Support swipe-up to advance to the next video, plus button and keyboard equivalents so the feature is not swipe-only.
- Build the initial queue from local similar-video candidates, excluding the seed video, videos already seen in the feed session, unsupported/unplayable videos, and NSFW videos unless NSFW visibility is enabled.
- Record feed-session events with enough data to classify feedback: session ID, video ID, queue position, playback-start state, watch duration, video duration, completion ratio, action, and timestamp.
- Treat quick swipe-away as a soft negative only when playback actually started and both dwell time and completion ratio are low, for example `<3s` and `<20%` watched.
- Treat watch-through or high completion as positive feedback, based on completion ratio or natural end, for example `>=70-80%` watched.
- Re-rank only future unseen items after feedback; do not disrupt the current video or immediately preloaded next video.
- Keep the adaptive state session-local for the MVP unless persistent history is designed with explicit clear/reset controls.
- Prevent runaway narrowing by applying repeat suppression, near-duplicate limits, and some diversity/exploration when the candidate pool is large enough.
- Handle queue exhaustion with a clear end state and an action to return to browsing or restart from another seed.
- Preserve mobile performance by preloading only the next small number of videos and releasing old video elements/sources on advance.
- Respect accessibility and motion preferences: focus trap while active, `Escape` exit, arrow-key next/previous, visible controls for gestures, and reduced-motion behavior.

## Acceptance Criteria

- A user can start the feed from any indexed video and gets a playable queue of similar videos.
- Swipe-up advances to the next queued video without page scroll leaking through the fullscreen feed.
- Current and next videos remain stable while feedback recalculates later queue positions.
- Fast skip classification requires confirmed playback plus low watch time and low completion ratio.
- High-completion classification records a positive signal and influences later unseen queue ordering.
- Already-seen videos do not repeat in the same session unless the candidate pool is exhausted.
- The queue avoids immediate near-duplicates when enough alternatives exist.
- The feed works on mobile Safari/Chrome constraints: muted autoplay, `playsinline`, and no forced native fullscreen takeover.
- The user can exit the feed at any time and returns to the prior browsing context.
- The feed has button/keyboard alternatives for next, previous, pause/play, and exit.
- UI smoke or browser tests cover mobile feed entry, swipe/next behavior, feedback classification, and queue exhaustion.
- Unit tests cover signal classification for short videos, long videos, playback-not-started skips, watch-through, and re-rank stability.

## Notes

- Public TikTok documentation says its For You system uses user interactions such as likes/shares/comments, videos watched in full, skips, watch time, and explicit `Not interested` feedback; it also deliberately diversifies recommendations and avoids repetitive patterns.
- YouTube's public analytics documentation exposes a Shorts metric called `Stayed to watch`, defined as the percentage of times viewers viewed Shorts versus swiped away, alongside average view duration and watch time.
- The Kuaishou CIKM 2023 paper on industrial short-video recommenders models implicit negative feedback from skipping behavior and defines `Glance Video Viewing` as watch time `<3 seconds`, while also using stronger positive watch-time signals. This supports using `<3s` as a useful signal, but not as the only signal.
- Advisor consensus: use session-based adaptation for the MVP, not persistent global preference mutation; use completion ratio as the primary signal; treat negative feedback as a soft penalty; keep the current and next item stable; prevent feed flapping and near-duplicate collapse.
- Candidate MVP algorithm: start with a fixed candidate pool from the seed video, maintain session positive/negative event lists, score remaining candidates by seed similarity plus recent-positive similarity minus soft negative similarity, then apply seen and diversity penalties.
- Candidate future enhancement: persist optional local watch events for long-term personalization, with clear history deletion and private/incognito mode.

## Implementation Progress

- Added `GET /api/search/similar-videos` as a stateless local candidate endpoint that seeds from an indexed video frame, filters seen/seed videos, dedupes by video, respects NSFW visibility, and returns video metadata for the feed UI.
- Added video-card and video-result `Feed` actions that open a fullscreen raw muted `playsinline` video overlay with button, keyboard, and swipe navigation plus a clear end state.
- Kept adaptation session-local in browser memory: feed events carry a session ID, queue position, watch/dwell/completion metrics, action, timestamp, and classified signal; quick skips require playback plus low dwell/completion, high completion/natural end is positive, and reranking only affects later unseen queue items.
- Added Playwright smoke coverage for mobile feed entry, page locking, feedback classification, next/keyboard navigation, queue exhaustion, and exit restoration, plus Go coverage for the similar-video endpoint and indexed seed-frame selection.
- Remaining before archiving: decide whether the broader acceptance criteria need stronger near-duplicate/diversity handling and dedicated JS unit tests beyond the smoke classifier assertions.

## References

- https://newsroom.tiktok.com/en-us/how-tiktok-recommends-videos-for-you
- https://support.tiktok.com/en/using-tiktok/exploring-videos/how-tiktok-recommends-content
- https://support.google.com/youtube/answer/9314355?hl=en-GB&co=GENIE.Platform%3DDesktop
- https://fi.ee.tsinghua.edu.cn/~gaochen/papers/CIKM2023.pdf
