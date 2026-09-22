# Clarify what “Exact only” means

## Priority

P1 — Design recommendation.

## Summary

The words “Exact only”, “copies”, and “Keep largest” make a perceptual-hash match feel more certain than it is.

## Evidence

The production view offers Exact only and labels groups as copies. In the first captured group, all five images are 450×424, but only the first is labelled largest. There is no visible explanation of ties or quality. The established feature contract is dHash equality/distance—not byte or pixel equality.

Reproduction / inspection steps:

1. Open Duplicates and select Exact only.
2. Inspect a group with equal dimensions and the keeper label.

## Risks

This is a destructive workflow. Hash equality, visual similarity, higher resolution, and better quality are different claims. The current labels blur them together.

## Requirements

- Rename Exact only to Identical visual hash; call the groups Possible matches. Explain that colors, edits, and fine detail may differ.
- Show an explicit Keep badge and let the user select another keeper. For equal dimensions say Same resolution, rather than implying one is larger.
- Add file size/format and an accessible side-by-side comparison. On phones, offer a two-up/flip comparison instead of only a long vertical list.
- Show the separate unattended-import policy/status when that feature is exposed in UI; do not imply the new import option fixes manual cleanup.

## Acceptance Criteria

- [ ] No label claims byte equality without verification; users can identify the keeper and understand the deletion/retention policy before confirming.
- [ ] Add red→green coverage for behavior changes; verify desktop and mobile UI.

## Notes

- Audit only; no fix implemented.
- [HTML report, UX-02](../../docs/reviews/2026-09-22-atelier-ux-audit.html#UX-02) contains embedded screenshot evidence.
