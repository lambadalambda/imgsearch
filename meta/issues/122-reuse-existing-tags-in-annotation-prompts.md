# Give the annotator the existing tag vocabulary

## Summary

Each annotation invents its own tags, so the library accumulates near-synonyms ("tabby", "tabby cat", "tabby-cat") and the tag cloud and tag search fragment. The model should see the tags the library already uses and prefer them when they fit.

## Requirements

- Load the most used tags (by media unit count, the same counting as the tag cloud) and pass them to the annotator as `KnownTags` on the image options and the video input.
- The shared prompts (image, video frame, video) tell the model to prefer those tags when they apply and to add new ones only for uncovered concepts. Both the native and the OpenAI-compatible backends get it automatically.
- Cache the list in the worker with a refresh interval so it is not recomputed per job; bound its size so the prompt stays small.

## Acceptance Criteria

- Prompt tests show the hint with the tags; worker tests show the fake annotator receiving the current top tags; a DB test covers the loader.
- No change when the library has no tags yet.

## Notes

- Reported 2026-09-22.
