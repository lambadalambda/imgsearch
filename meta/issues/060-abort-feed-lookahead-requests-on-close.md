# 060: Abort Feed Lookahead Requests on Close

## Priority

P2

## Status

Open.

## Summary

Atelier Feed invalidates stale lookahead responses with a token, but it does not abort in-flight HTTP requests when the overlay closes.

## Context

- `searchSimilarVideos` accepts an optional `AbortSignal`.
- `frontend/src/components/Feed.svelte` does not create or pass an `AbortController` for lookahead requests.
- Closing the Feed increments `candidateRequestToken`, so the stale response is ignored client-side, but server work continues.

## Risks

- Closing Feed can leave unnecessary search work running on the backend.
- Slow or repeated open/close interactions can waste CPU/DB/GPU resources.
- Future heavier similar-video ranking work could make this more visible.

## Acceptance Criteria

- [ ] Add coverage proving closing Feed aborts an in-flight similar-video fetch.
- [ ] Create an `AbortController` per Feed session or per lookahead request.
- [ ] Pass the signal to `searchSimilarVideos`.
- [ ] Abort on `close()` and when starting a new session.
- [ ] Ignore aborts without marking the Feed exhausted.

## Related Files

- `frontend/src/components/Feed.svelte`
- `frontend/src/lib/api.ts`
