# Share media action state between card and lightbox

## Summary

`Pin.svelte` (lines 28, 80-85, 127) keeps a local `nsfwLocal` override and never writes NSFW state back to the `pins` store, while `Lightbox.svelte` (lines 36-39, 51) reads and writes `pin.isNSFW` on the store. Flagging from the card, opening the lightbox, and clicking again therefore un-flags on the server while the card still shows flagged. Both components also duplicate `runAction`, target-id selection, optimistic NSFW, re-annotate, and delete-with-confirm (`Pin.svelte:55-108`, `Lightbox.svelte:18-73`).

## Requirements

- Extract `frontend/src/lib/mediaActions.ts` exposing `{ pending, error, flagNSFW, reannotate, remove }` built on the store.
- Both Pin and Lightbox use it; remove `nsfwLocal` and the redundant `isHidden` in Pin.
- Optimistic updates go to the store so every view of a pin agrees.

## Acceptance Criteria

- Smoke test: toggle NSFW from the card, open the lightbox, verify the lightbox shows the flagged state, toggle from the lightbox, verify the card updates.
- No duplicated action code between the two components.

## Notes

- Found during the 2026-09-21 review.
