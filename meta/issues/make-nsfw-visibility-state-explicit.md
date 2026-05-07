# Make NSFW visibility state explicit

## Summary

The `Show NSFW` checkbox is visually similar to ordinary utility controls despite controlling sensitive content visibility.

## Requirements

- Replace or augment the label with explicit state text such as `NSFW: Hidden` and `NSFW: Visible`.
- Use a stronger visual treatment when NSFW visibility is enabled.
- Preserve persisted preference and backend `include_nsfw` behavior.
- Consider whether a one-time confirmation or thumbnail blurring is needed.

## Acceptance Criteria

- Users can immediately tell whether NSFW content is hidden or visible.
- Toggling NSFW still refreshes image/video/search/tag requests with correct filtering.
- Enabled state is visually distinct from normal utility controls.
- Existing smoke coverage for `include_nsfw=1` still passes.

## Notes

- Keep the first pass small; confirmation/blurring can become follow-up work if needed.
