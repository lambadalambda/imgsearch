# Show annotation progress on cards and refresh them live

## Summary

While the annotator works (native or remote), the Atelier shows nothing per item: cards carry only the embed job state, which is not rendered, and the Atelier never subscribes to `/api/live`. A user who just switched backends or queued "Re-annotate all" cannot see which images are queued, which one is being annotated, which failed, or when new text has landed.

## Requirements

- Expose an `annotation_state` (`queued`, `annotating`, `failed`, `done`, `none`) and `annotation_updated_at` per image and video in the list responses.
- Add a cheap status endpoint for a set of visible ids that returns those fields plus the embed state.
- The Atelier polls it every few seconds for the pins on screen while the tab is visible, shows a badge on queued/annotating/failed cards, and reloads a card's title, summary, description, and tags in place when its annotation timestamp changes (the open lightbox follows).

## Acceptance Criteria

- Smoke test: a card shows "Annotating…" and then its title updates without a reload once the stub reports the annotation as done.
- The status query uses the `index_jobs` lookup indexes (no scans) and is one round trip per poll.

## Notes

- Reported 2026-09-22. Polling was chosen over the live WebSocket snapshot because it is per visible item, one indexed query, and needs no new push plumbing.
