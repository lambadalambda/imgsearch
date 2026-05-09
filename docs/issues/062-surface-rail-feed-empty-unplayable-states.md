# 062: Surface Rail Feed Empty and Unplayable States

## Priority

P2

## Status

Open.

## Summary

The Atelier Rail Feed button can silently do nothing when there are no videos, no browser-playable videos, or the random sampling attempts miss playable candidates.

## Context

- `frontend/src/components/Rail.svelte` samples `/api/videos` and tries up to six random offsets.
- If `total <= 0`, no record exists, or no sampled record passes `canPlayMime`, the function returns without user feedback.
- The button does expose a loading/busy state while selecting, but not an empty/error state.
- Decision: the Rail Feed launcher should stay globally random over `/api/videos`, independent of current search/similar context.

## Risks

- Users can click Feed and see no visible result or explanation.
- Libraries with many unplayable videos can make the feature feel broken.
- Support/debugging is harder because no state is surfaced.

## Acceptance Criteria

- [ ] Add smoke or component coverage for zero-video and no-playable-video responses.
- [ ] Surface a user-visible message when Feed cannot start.
- [ ] Consider disabling the Rail Feed button when stats report zero videos.
- [ ] Consider a broader or deterministic fallback scan after random attempts fail.

## Related Files

- `frontend/src/components/Rail.svelte`
- `frontend/src/lib/utils.ts`
- `scripts/ui_smoke_test_atelier.mjs`
