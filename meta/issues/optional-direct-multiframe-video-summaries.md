# Optional direct multi-frame video summaries for external APIs

## Summary

Add a saved external-API video input mode: synthesize existing compact frame descriptions (default) or summarize all stored sampled frame images directly in one multimodal request. Native annotation retains the memory-conservative existing path. New uploads currently default to five uniform samples; existing videos keep their stored samples.

## Requirements

- Keep video frame extraction, embeddings and timestamped search unchanged.
- Persist an explicit remote video mode and a bounded sampled-frame count in Settings with old-settings/omitted-field compatibility, validation, draft retention and honest capability/resource guidance. Frame count defaults to five and affects new uploads only; existing videos are not resampled.
- Direct mode skips per-frame description generation, supplies chronological images with timestamps and optional filename/duration/transcript context, and retains all image evidence on compact/API-format retries.
- Native annotation and the default description pipeline remain unchanged. Settings changes affect subsequent jobs, never automatically requeue existing media.
- Preserve conservative NSFW behavior for videos and sampled-frame search without inventing per-frame classification evidence.
- TDD, independent review, full checks and topical commits. No production configuration changes, live reannotation, deployment or push requested.

## Acceptance Criteria

- [ ] Settings roundtrip/default/validation and runtime mode switching are tested.
- [ ] Fake remote requests prove all chronological images are sent on primary/retry paths; single-image and native paths stay compatible.
- [ ] Worker tests prove direct mode skips frame descriptions while preserving embeddings, frame paths/timestamps and safe NSFW handling.
- [ ] Desktop/mobile Settings UI, documentation and full tests pass; reviewed and committed.

## Notes

- Related broader performance work: [073](073-reduce-video-annotation-work.md). This option does not claim measured quality/speed improvement or universal provider multi-image support.
