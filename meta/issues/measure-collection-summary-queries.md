# Measure collection summary queries at representative scale

## Summary

UX-07 summaries use one set-based query and return at most four covers per collection, but counts, visibility checks and ranking still process saved memberships and relevant video frames. Independent review found no correctness or evident performance blocker; representative measurement is a follow-up, not a reason for speculative optimization.

## Acceptance Criteria

- Inspect EXPLAIN QUERY PLAN and benchmark representative collection/membership/frame distributions, including one video saved in multiple populated collections.
- Verify independent per-collection counts and cover ordering for shared videos.
- Record scaling and only propose changes if measured cost warrants them; preserve filter-before-cover privacy and stable ordering.

## Notes

- Index summary work only. Existing collection-detail video hydration optimization remains a separate issue.
- No performance threshold or regression claim established by the source-only review.
