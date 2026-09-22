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

- [x] Settings roundtrip/default/validation and runtime mode switching are tested.
- [x] Fake remote requests prove all chronological images are sent on primary/retry paths; single-image and native paths stay compatible.
- [x] Worker tests prove direct mode skips frame descriptions while preserving embeddings, frame paths/timestamps and safe NSFW handling.
- [x] Desktop/mobile Settings UI, documentation and full tests pass; reviewed and committed.

## Notes

- Related broader performance work: [073](073-reduce-video-annotation-work.md). This option does not claim measured quality/speed improvement or universal provider multi-image support.
- `a1622d4` adds persisted mode/count, strict 1–32 API/CLI sampling validation, legacy flag compatibility and upload-time settings snapshots. Existing/duplicate videos are not resampled; settings saves do not create/reset jobs.
- Direct external requests carry all stored samples, including ten-frame legacy videos and repeated images at distinct timestamps. Primary, compact parsing retry, response-format fallback and transient transport retry retain image/timestamp evidence and video context. Missing images or provider rejection fail explicitly; default/native descriptions stay intact.
- The resolver acquires one immutable backend/mode/token-budget snapshot for the complete video job, including frame preparation. Direct mode does not write frame annotations or embeddings. Video-level any-frame NSFW classification uses the existing tag parser/merge; single-video reannotation retains old metadata so its only NSFW signal is not cleared while pending/failed.
- Settings UI retains native/external mode preference across saved backend switches and reloads. Numeric invalid/blank values block submission; both fields survive navigation, save failures and discard. Desktop1440/mobile390 screenshot inspection confirmed readable stacked controls and no horizontal overflow. The test fixture now preserves stored key presence on blank-key saves, matching the existing server contract.
- TDD red evidence: missing/default API fields and invalid values incorrectly accepted; absent upload callback and count seed; CLI33 accepted; undefined draft fields and missing browser controls; direct-input API/compiler failures, unpinned worker routing and reannotation clearing metadata. Green: full Go/frontend/script/browser suite (67 frontend tests), Svelte check0 errors/warnings and focused race suites for settings/upload/app/cmd/worker/annotation/remote/videos.
- Independent sampling/settings, UI, backend and final integration reviews found no remaining blockers. Added regression coverage for saved legacy sampling overrides, unchanged existing frame/job rows, saved-native mode retention and pinned per-video budgets.
- User/operator guide: [video summaries](../../docs/video-summaries.md); architectural rationale: ADR-013. No production deployment/settings edits/reannotation or Git push performed. Real-provider multi-image acceptance and comparative quality/performance remain unverified; tests use controlled local HTTP fixtures.

Completed and archived. Backend direct mode is committed in `82bfa63`; Settings/UI is included in the final topical UI commit.
