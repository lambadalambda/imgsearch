# Simplify masthead search and filter controls

## Summary

The masthead packs brand, search input, exclude controls, search, upload, and overflow actions into one dense row. This weakens search as the primary action and makes filtering state easy to miss.

## Requirements

- Rename the `Exclude` disclosure to a broader `Filters` or `Refine` label.
- Show an active filter count when negative text or tag filters are set.
- Show active filter chips near the search input when the filter panel is closed.
- De-emphasize secondary actions such as upload and overflow compared with search.
- Keep desktop and mobile wrapping predictable.

## Acceptance Criteria

- Users can tell when filters are active without opening the filter panel.
- The search field and search button remain the dominant controls in the masthead.
- Upload and library utility actions no longer compete visually with the search CTA.
- Mobile search controls avoid awkward multi-line clutter.

## Notes

- This issue may be split further if the masthead/layout changes become large.
