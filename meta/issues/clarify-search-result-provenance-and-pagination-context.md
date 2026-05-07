# Clarify search result provenance and pagination context

## Summary

The results tab explains that results persist, but it does not make the current query, filters, or search type obvious. Pagination labels also rely on users knowing the sort order.

## Requirements

- Show what produced the current results: text query, tag search, or similar-image search.
- Include active negative query and tag filters where relevant.
- Make result count and clear/refine actions easy to find.
- Clarify `Newer`/`Older` pagination with sort context.

## Acceptance Criteria

- Search results show a concise provenance bar such as `Results for: ...`.
- Tag and similar-image result modes identify their source clearly.
- Pagination labels make the sort direction understandable.
- Existing result pagination behavior remains unchanged.

## Notes

- This can share state with the masthead filter chips issue if implemented together.
