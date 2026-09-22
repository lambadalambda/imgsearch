# Make Collections feel like saved media

## Priority

P2 — Design recommendation.

## Summary

The collection index resembles a settings list: names and management buttons, but no visual memory cues or counts.

## Evidence

Populated and empty collections look alike in the list. Rename/Delete occupy much of each row. The empty-detail state says items might be NSFW-hidden even for a genuinely empty list, and offers no direct Browse library action. Create, save, rename and delete worked in the sandbox.

Reproduction / inspection steps:

1. Open Collections with one populated and one empty list.
2. Open the empty list and inspect its next-step guidance.

## Risks

Collections are a visual recall tool. Without a cover or count, users must remember what every name contains; management receives more emphasis than opening content.

## Requirements

- Add a small cover mosaic and item count; make the whole card a clearly clickable browse target.
- Move Rename/Delete into a labelled overflow menu while keeping delete confirmation and the promise that originals are retained.
- Distinguish No saved items yet from Items hidden by filters, with Browse library or Show hidden items as appropriate.
- Keep saved ordering stable and show short confirmation when a picker adds/removes an item.

## Acceptance Criteria

- [ ] A user can distinguish populated/empty/filtered collections at a glance and reach the next useful action directly.
- [ ] Add red→green coverage for behavior changes; verify desktop and mobile UI.

## Notes

- Audit only; no fix implemented.
- [HTML report, UX-07](../../docs/reviews/2026-09-22-atelier-ux-audit.html#UX-07) contains embedded screenshot evidence.
