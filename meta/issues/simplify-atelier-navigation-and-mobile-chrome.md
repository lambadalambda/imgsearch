# Give mobile navigation fewer jobs

## Priority

P1 — Design recommendation.

## Summary

A phone gets the entire desktop rail as a row of tiny icons, followed by several more rows of controls before any media.

## Evidence

At 390×844, the first media begins around y=373—44% of the viewport. Navigation buttons measure about 29×40 CSS pixels. Library and Collections use near-identical grid icons; disabled Tags and the legacy link consume space. Settings/Statistics/Collections also inherit the full search-and-tag band.

Reproduction / inspection steps:

1. Open the library at 390×844 and compare the space above the first image.
2. Open Settings at the same width; note the same search/tag band above the form.

## Risks

The visual system is appealing, but users must decode the icons, remember their destinations, and navigate small targets. Utility pages look like another library mode rather than a focused task.

## Requirements

- Use Library, Search, Collections and More as the primary mobile navigation; move Statistics, Duplicates and Settings under More.
- Use persistent desktop labels and a distinct Collections icon. Remove Tags (soon) until useful and retire the legacy escape link.
- Collapse media/sort/NSFW options behind one clearly labelled filter control on small screens; retain a visible active-filter summary.
- Use a compact utility-page header instead of repeating global tags/counts. Aim for comfortable 44px touch targets; this audit is not a formal WCAG conformance test.

## Acceptance Criteria

- [x] Labelled core navigation with 44px touch targets and first media within the top third of a 390×844 viewport, without losing filter discoverability.
- [x] Add red→green coverage for behavior changes; verify desktop and mobile UI.

## Notes

- Implemented in the main checkout; awaiting parent review. No commit, index/archive update, or deployment in this task.
- Mobile uses a fixed bottom Library / Search / Collections / More navigation with safe-area padding and labelled targets at least 44×44px. More exposes Settings, Statistics, Duplicates **and Feed**; Escape closes it and restores focus. Desktop keeps persistent labels and gives Collections a folder icon.
- Search navigates out of utility/collection pages and focuses the existing entry, rather than creating a new backend mode. Upload remains in every page header.
- Mobile Filters collapses type/sort/NSFW controls while retaining an active scope summary. Browse tags is a separate disclosure. Settings/Statistics/Duplicates/Collections and collection detail have no global search/tag band; collection NSFW controls and detail back navigation remain.
- The App no longer makes a hidden library request on utility pages. Collection request identities, counts, error handling, and filter race guards are unchanged.
- TDD: new browser assertions failed on the old shell (missing scoped totals); pure helper tests initially failed before implementation. The final local fixture smoke verifies persistent labels, distinct icons, keyboard More/Search navigation, Feed/Upload reachability, collapsed controls, and overflow at 320/390/640/768/1280px. The 390×844 first-media measurement is **y=241.8px (28.6%)**, below the 281.3px target.
- Review correction: choosing a mobile tag or Browse library restores focus to the visible Browse tags disclosure before collapsing its controls. Desktop keeps focus on the still-visible shortcut. A browser regression failed on the previously hidden focused tag and now verifies both mobile actions.
- Validation: `mise run test:full`, `mise run check:frontend` (zero warnings/errors), formatting of new/reworked UI/helpers/tests, and `git diff --check`. Existing UX07 collection count/error/filter-race smoke tests pass with only label/layout selector adjustments.
- Inspected local fixture screenshots in `/tmp/ux03-04/` (library desktop/mobile and compact utility header). Fixtures use neutral media placeholders; no production browser was used. This is layout/behavior verification, not formal accessibility conformance.
- [HTML report, UX-03](../../docs/reviews/2026-09-22-atelier-ux-audit.html#UX-03) contains embedded screenshot evidence.

## Completion

Independent source review approved after delayed/failed/stale scope regressions and tag-focus fixes. Parent full tests and frontend checks passed; first media y=241.8 at390x844, core targets44px. Archived; broader card touch targets and live annotation classification visibility are not claimed fixed.
