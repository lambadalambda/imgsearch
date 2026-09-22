# Protect unsaved Settings edits

## Priority

P2 — Reproduced workflow issue.

## Summary

A simple navigation click silently throws away the Settings draft.

## Evidence

In the sandbox, changed the native variant from e4b to 26b without saving, navigated to Library, then back to Settings: e4b returned and Save was disabled. Repeated via Collections. There was no discard warning or retained draft. No model configuration was saved.

Reproduction / inspection steps:

1. Settings → change a field; do not save.
2. Choose Library or Collections, then return to Settings.
3. The prior value is restored without warning.

## Risks

The same behavior can cost a longer remote-server configuration edit. Global search and tag links remain prominent while editing, making accidental departure easy.

## Requirements

- Add a dirty-state guard with Stay / Discard changes, or retain an explicit unsaved draft across internal navigation.
- Keep a Save changes / Discard bar near edited controls, and show a clear Saved confirmation.
- Related layout improvement: avoid equal-height Settings cards that create a large empty backend panel; use independent panels or a compact form layout.

## Acceptance Criteria

- [ ] Leaving a dirty form cannot silently erase edits; the user can distinguish draft, saved, and active worker configuration.
- [ ] Add red→green coverage for behavior changes; verify desktop and mobile UI.

## Notes

- Audit only; no fix implemented.
- [HTML report, UX-05](../../docs/reviews/2026-09-22-atelier-ux-audit.html#UX-05) contains embedded screenshot evidence.
