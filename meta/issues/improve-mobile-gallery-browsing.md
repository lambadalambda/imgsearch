# Improve mobile gallery browsing

## Summary

The mobile gallery currently reads like a one-card-per-row metadata list. The screenshots showed that this wastes browsing space and makes tabs and actions feel cramped.

## Requirements

- Use a denser mobile gallery layout that shows more image content per screen.
- Hide or reduce secondary metadata on narrow screens.
- Shorten mobile tab labels, especially `Search Results`.
- Ensure important touch targets are at least approximately 40-44px tall.
- Keep mobile lightbox padding small enough that images remain prominent.

## Acceptance Criteria

- A 390px-wide viewport can browse multiple cards per screen comfortably.
- Mobile card titles, tags, and actions do not overflow horizontally.
- Tabs remain understandable without severe truncation.
- Touch interactions remain accessible on coarse-pointer devices.

## Notes

- Candidate approach: two-column gallery on small screens with filename/path/status hidden or minimized.
