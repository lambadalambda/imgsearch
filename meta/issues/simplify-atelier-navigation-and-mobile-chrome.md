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

- [ ] Proposed target to agree during implementation: labelled core navigation with 44px touch targets and first media within the top third of a 390×844 viewport, without losing filter discoverability.
- [ ] Add red→green coverage for behavior changes; verify desktop and mobile UI.

## Notes

- Audit only; no fix implemented.
- [HTML report, UX-03](../../docs/reviews/2026-09-22-atelier-ux-audit.html#UX-03) contains embedded screenshot evidence.
