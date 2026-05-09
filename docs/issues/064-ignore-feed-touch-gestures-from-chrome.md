# 064: Ignore Feed Touch Gestures from Chrome Controls

## Priority

P3

## Status

Open.

## Summary

Atelier Feed touch handlers are attached to the overlay and can interpret drags beginning on controls as feed swipes. Boundary swipe velocity is also computed from the rubber-banded visual offset instead of raw movement.

## Context

- `frontend/src/components/Feed.svelte` handles touch start/move/end on the overlay.
- `onCanvasClick` ignores `[data-feed-chrome]`, but touch start does not.
- At queue boundaries, `dragOffsetPx` is multiplied by the rubber-band factor and then used for velocity calculation.

## Risks

- Dragging on buttons or progress chrome can accidentally navigate the Feed.
- Fast boundary swipes can feel weaker than expected because velocity is damped by the visual rubber-band offset.
- Mobile Feed behavior can drift from the legacy thresholds.

## Acceptance Criteria

- [ ] Add mobile/viewport smoke coverage for touch interactions that start on Feed chrome.
- [ ] Ignore touch gestures that begin inside `[data-feed-chrome]` or interactive controls.
- [ ] Track raw `dy` separately from rubber-banded visual offset and use raw movement for velocity.
- [ ] Preserve existing swipe thresholds and click suppression behavior.

## Related Files

- `frontend/src/components/Feed.svelte`
- `scripts/ui_smoke_test_atelier.mjs`
