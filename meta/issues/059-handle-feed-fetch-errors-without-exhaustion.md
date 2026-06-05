# 059: Handle Feed Fetch Errors Without Ending the Session

## Priority

P2

## Status

Open.

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

- [ ] Add smoke or component-level coverage for a transient similar-video fetch error.
- [ ] Keep true empty successful responses as the only normal exhaustion signal.
- [ ] Show a retryable error state or allow the next lookahead attempt to retry after transient failures.
- [ ] Do not append stale results from old sessions after an error.

## Related Files

- `frontend/src/components/Feed.svelte`
- `frontend/src/lib/api.ts`
- `scripts/ui_smoke_test_atelier.mjs`
