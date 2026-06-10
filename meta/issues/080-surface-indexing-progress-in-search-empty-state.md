# Surface indexing progress in the search empty state

## Summary

A text search against a library whose embeddings are still being computed returns "No matches for …" — indistinguishable from a genuine miss. Observed live: query returned 0 results while stats reported 0 / 51 images embedded. This is exactly the moment a new user decides whether search "works"; the client already holds `$stats` and can tell the difference.

## Requirements

- When a search/similar query returns zero results and the embed queue shows incomplete progress (`done < expected`), the empty state explains that indexing is still in progress and shows the progress (e.g. "Library is still indexing — 0 of 51 images embedded. Results will improve as indexing completes.").
- Genuine no-match libraries (fully embedded) keep the current message.
- Stats used for the hint should be reasonably fresh (refetch on search, or reuse issue 081's refresh).

## Acceptance Criteria

- With a stub stats payload reporting incomplete embedding, an empty search shows the indexing hint; with complete embedding it shows "No matches…".
- Smoke test covers both messages.

## Notes

- `frontend/src/App.svelte` (`emptyMessage`), `frontend/src/lib/stores.ts` (`stats`).
- Found during the 2026-06-10 UI/UX review.
