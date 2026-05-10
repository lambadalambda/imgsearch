# 061: Port Feed Buffer Reranking or Document the Divergence

## Priority

P2

## Status

Completed.

## Summary

The legacy Feed reranked already-buffered future queue items after feedback. The Atelier Feed only sends derived tag preferences to later fetches, so existing buffered candidates do not adapt.

## Context

- Legacy `internal/webui/static/app.js` reranks `queue[currentIndex+2...]` after feedback using visual distance plus a small tag-affinity nudge.
- Atelier `frontend/src/components/Feed.svelte` updates session tag scores but does not reorder buffered queue items.
- The difference is user-visible in longer sessions where several candidates are already queued.
- Decision: port the legacy in-buffer reranking behavior to Atelier rather than documenting a divergence.

## Risks

- Feed adaptation feels slower than the legacy implementation.
- Session feedback only affects future batches, not the current buffered queue.
- Documentation can imply stronger parity than the port currently provides.

## Acceptance Criteria

- [x] Decide whether in-buffer reranking is required for Atelier parity.
- [x] If required, add a pure or component-level test for reranking `queue[currentIndex+2...]` after feedback.
- [x] Preserve the current and immediately preloaded next item to avoid playback instability.
- [x] Mirror or intentionally revise the legacy scoring formula.
- [x] If not required, document the deliberate divergence in `docs/frontend.md`.

## Resolution

- Atelier now reranks already-buffered future items starting at `currentIndex + 2` after positive or soft-negative tag feedback.
- The scoring mirrors legacy: visual score plus a clamped tag-affinity nudge.
- Current and immediately preloaded next items remain stable.
- Browser smoke coverage verifies a buffered future item with matching feedback tags is promoted while the current/next path remains stable.

## Related Files

- `frontend/src/components/Feed.svelte`
- `frontend/src/lib/feed.ts`
- `internal/webui/static/app.js`
- `docs/frontend.md`
