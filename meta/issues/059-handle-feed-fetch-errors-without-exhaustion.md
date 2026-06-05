# 059: Handle Feed Fetch Errors Without Ending the Session

## Priority

P2

## Status

Resolved (2026-06-05). Transient `/api/search/similar-videos` failures now show a retryable Feed error state instead of marking the session exhausted. Covered by the Atelier smoke test.

## Summary

Atelier Feed currently treats any `/api/search/similar-videos` request error as feed exhaustion. Transient failures should not present as a true end-of-feed state.

## Context

- `frontend/src/components/Feed.svelte` sets `exhausted = true` in the catch block for `searchSimilarVideos`.
- A 429, 500, 503, network interruption, or aborted request is semantically different from a successful empty candidate response.
- The UI displays the exhausted state as "That's the end of this feed.".

## Risks

- Temporary backend errors end the user's session.
- Users receive misleading feedback that no candidates exist.
- Retry behavior is not available without closing and reopening Feed.

## Acceptance Criteria

- [x] Add smoke or component-level coverage for a transient similar-video fetch error.
- [x] Keep true empty successful responses as the only normal exhaustion signal.
- [x] Show a retryable error state or allow the next lookahead attempt to retry after transient failures.
- [x] Do not append stale results from old sessions after an error.

## Resolution

- `Feed.svelte` now stores similar-video fetch failures in a retryable error state and leaves `exhausted` reserved for successful empty responses.
- Retry reuses the existing lookahead path, so successful retry batches append normally.
- Session token checks now also guard failed requests, preventing old-session failures from mutating a newer Feed session.

## Related Files

- `frontend/src/components/Feed.svelte`
- `frontend/src/lib/api.ts`
- `scripts/ui_smoke_test_atelier.mjs`
