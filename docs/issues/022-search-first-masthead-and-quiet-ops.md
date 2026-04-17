# 022: Shift to a Search-First Masthead and Quiet Ops Controls

## Status: Completed

## Goal

Make the top chrome feel like a compact tool header instead of a hero/dashboard by keeping Search as the only dominant action and demoting occasional operations (upload, retry, refresh, indexing diagnostics).

## Problem

The previous masthead carried oversized brand treatment, explanatory copy, and a full indexing status block in the same visual zone as search. That made the top area feel too heavy and pushed primary browsing/search interactions down the page.

## Desired Direction

- Treat search as the primary persistent action.
- Keep upload and indexing operations discoverable but visually quiet.
- Preserve all functionality and accessibility semantics.
- Keep desktop and mobile behavior stable.

## Scope

- Refactor masthead layout to compact app-chrome proportions.
- Move indexing controls/status into a quieter ops bar.
- Demote optional Exclude input behind an advanced disclosure.
- Keep retry/refresh available via a low-prominence disclosure menu.
- Add/adjust regression tests for updated HTML/CSS/JS structure.

## Scope Landed

- Removed heavyweight masthead panel treatment and hero-style heading scale.
- Reworked masthead into a compact row with brand + search form + quiet upload action.
- Moved Exclude input behind a `search-advanced` disclosure so base search stays focused.
- Replaced the old status strip with an `ops-bar` utility row and a quiet `ops-menu` disclosure for Retry stuck / Refresh.
- Added JS ops-bar state toggling (`active` vs `idle`) so non-essential status detail/progress can stay visually quieter when idle.
- Preserved all existing IDs and behaviors for search, upload modal, stats refresh, and retry actions.

## Acceptance Criteria

- [x] Search is the clear primary action in the top chrome
- [x] Upload/retry/refresh remain available but visually de-emphasized
- [x] Indexing status remains readable without dominating the header
- [x] Compact layout works on desktop and mobile breakpoints
- [x] Web UI regression tests cover the new structure and behavior hooks

## Notes

- This issue follows feedback from a second UX review focused on reducing masthead prominence.
- Further tuning can iterate on exact spacing/typography values without changing the structural direction.
