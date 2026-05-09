# 056: Reset Atelier Pagination on Data Key Changes

## Priority

P1

## Status

Open.

## Summary

After `Load more` is used once, the Atelier data effect treats future fetches as append operations because `$pageBump > 0` stays true. Mode changes, NSFW toggles, and upload refreshes can append new results into old pins with stale offsets.

## Context

- `frontend/src/App.svelte` uses `$pageBump` to decide whether the next fetch appends.
- `bumpPage()` only increments the store; no code consumes or resets the bump for a specific data key.
- The effect also depends on mode, NSFW state, and `dataEpoch`, but still appends if `pageBump` is greater than zero.

## Risks

- Search, tag, similar, and library results can mix in the same masonry after a previous Load More.
- Upload refreshes can append from a stale offset instead of replacing with the first page.
- NSFW toggles can skip the first page and show inconsistent totals.

## Acceptance Criteria

- [ ] Add Atelier smoke or unit coverage for changing mode after Load More.
- [ ] Add coverage for an upload/data refresh after Load More replacing the collection.
- [ ] Track the last handled page bump separately from the current data key.
- [ ] Reset `currentOffset` and replace pins whenever mode/query/tags/NSFW/dataEpoch changes.
- [ ] Preserve append behavior only for an actual new Load More request on the same data key.

## Related Files

- `frontend/src/App.svelte`
- `frontend/src/lib/stores.ts`
- `scripts/ui_smoke_test_atelier.mjs`
