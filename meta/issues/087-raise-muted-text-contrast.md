# Raise muted text contrast to WCAG AA

## Summary

`--color-muted-2` (#948b7e) on the app background (#faf7f2) is roughly 3:1 contrast — below the WCAG AA 4.5:1 requirement for normal-size text. It is used for small text throughout: result counts, search placeholder, file sizes, stats captions, card metadata.

## Requirements

- Darken `--color-muted-2` (and audit `--color-muted`) so body-size text meets 4.5:1 against `--color-bg` and `--color-surface`.
- Keep the warm hue; this is a value tweak, not a palette change.

## Acceptance Criteria

- Computed contrast of muted-2 text on bg/surface ≥ 4.5:1.
- Visual spot-check that the hierarchy (ink > muted > muted-2) still reads correctly.

## Notes

- `frontend/src/app.css:15-16`.
- Found during the 2026-06-10 UI/UX review.
