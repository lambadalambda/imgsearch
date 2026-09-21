# De-duplicate pins by key on Load More

## Summary

`App.svelte:250-251` appends the next page's pins without checking for keys already present. Library `newest` sort and text/tag search use offset pagination, so if the indexer finishes an item between page 1 and Load More the pages shift and a key repeats. Svelte 5's keyed `{#each}` throws `each_key_duplicate` in production builds, which takes down the whole grid.

## Requirements

- Filter appended pins against the keys already in the store before appending.
- Advance `currentOffset` by the number of pins fetched, not the number kept, so the next page is not re-requested.

## Acceptance Criteria

- Smoke test: stub returns an overlapping item on page 2; the grid renders without error and shows the item once.
- No `each_key_duplicate` in the console during the smoke run.

## Notes

- Similar mode is unaffected because the backend sets `total = len(results)` (`internal/search/http.go:301`).
- Found during the 2026-09-21 review.
