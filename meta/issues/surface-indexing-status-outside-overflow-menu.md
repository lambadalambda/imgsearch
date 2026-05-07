# Surface indexing status outside the overflow menu

## Summary

Connection and indexing status are currently buried inside the top-right overflow menu even though library freshness is central to this app.

## Requirements

- Show a compact connection/indexing summary outside the overflow menu.
- Keep detailed progress, retry actions, and failure details inside the menu.
- Use clear status text and state coloring without making the masthead noisy.
- Avoid duplicating noisy screen-reader live announcements.

## Acceptance Criteria

- Users can see whether live updates are connected without opening the overflow menu.
- Users can see whether indexing is idle, active, or failing at a glance.
- Existing retry/refresh/failure controls remain available.
- Screen-reader announcements remain concise.

## Notes

- Both reviewer agents independently called out buried status as a usability issue.
