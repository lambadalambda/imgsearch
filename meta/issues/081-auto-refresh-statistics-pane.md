# Auto-refresh the Statistics pane

## Summary

Stats are fetched once at app bootstrap; the Statistics pane renders progress bars and queue counts that never change until a full page reload. As an indexing-progress dashboard it should update itself.

Also: the failures card header reads "Last 0 jobs that exhausted retries" when there are none.

## Requirements

- While the stats view is open, poll `/api/stats` (and the tag cloud, cheaply) on an interval (a few seconds) and update the pane in place.
- Stop polling when the view is left or the tab is hidden (`document.visibilitychange`).
- Fix the failures-card copy for the zero case (e.g. just "Recent failures" with the existing "No recent failures." body).

## Acceptance Criteria

- With a stub server that increments job counts per request, the rendered numbers change without a reload while `?view=stats` is open.
- Polling stops after navigating back to the library (no further /api/stats requests).
- Zero-failure copy no longer says "Last 0 jobs".

## Notes

- `frontend/src/components/StatisticsPane.svelte`, bootstrap fetch in `frontend/src/App.svelte:85-92`.
- Found during the 2026-06-10 UI/UX review.
