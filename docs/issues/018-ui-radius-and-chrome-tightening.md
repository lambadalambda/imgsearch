# 018: Tighten UI Radius and Chrome Hierarchy

## Status: Completed

## Goal

Make the web UI feel more elegant and less amateurish by tightening the radius system, reducing soft "pill" shapes, and simplifying the overall chrome hierarchy without changing the app architecture.

## Problem

The current UI uses too many rounded shapes at too many sizes:

- large panel radii
- rounded cards
- rounded inputs
- pill-shaped buttons
- pill-shaped tab counts
- pill-shaped state chips

This gives the interface a soft template-like feel rather than a tighter, more editorial product feel.

## Desired Direction

- Collapse the radius language into two main scales:
  - one for large containers/panels
  - one for interactive controls/cards
- Reserve fully pill-shaped elements only for true dot/progress affordances.
- Reduce visual chrome in the default state:
  - less visible framing around tabs
  - quieter count badges
  - less heavy control surfaces

## Scope

- Adjust root radius tokens in `internal/webui/static/styles.css`
- Reduce over-rounded buttons, inputs, cards, tabs, and chips
- Simplify tab-count presentation
- Revisit rest-state shadow and border intensity for cards/panels

## Scope Landed

- Replaced the old three-step radius token system with two shared scales:
  - `--radius-panel`
  - `--radius-control`
- Rebased buttons, inputs, tabs, cards, status chips, modal surfaces, and lightbox controls on the tighter shared radius language.
- Reduced heavy chrome in tab controls and selected-tab framing.
- Softened rest-state panel/card shadow and border treatment while preserving focus/hover contrast.
- Kept full pill radius for intentional dot/progress affordances.

## Acceptance Criteria

- [x] Buttons, cards, inputs, and tabs share a smaller, more consistent corner language
- [x] Pill-shaped treatment is limited to a small set of intentional UI elements
- [x] The UI reads visually calmer and less like a generic app template
- [x] No layout or behavior regressions in the existing web UI/server tests

## Notes

- This issue should focus on the visual language only, not card-content restructuring.
- Card layout and hover/overlay behavior are tracked separately.
