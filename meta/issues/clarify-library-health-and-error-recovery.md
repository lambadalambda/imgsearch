# Turn operational states into next steps

## Priority

P2 — Design recommendation.

## Summary

Scheduler vocabulary and generic errors tell users something happened, but not what it means or how to recover.

## Evidence

Production Statistics shows queued and ready counts together, plus 468 videos needing a transcription job with none queued. It does not explain whether transcription is unavailable or simply incomplete. An injected local search-network failure displayed Failed to fetch twice, without Retry.

Reproduction / inspection steps:

1. Open Statistics and compare the queued/ready/processed labels and denominators.
2. In a disposable environment, interrupt a search request and inspect the resulting message.

## Risks

“Ready” sounds completed to a non-operator. An empty progress bar without an explanation can look broken even when a feature is intentionally disabled. Raw fetch errors do not offer a recovery path.

## Requirements

- Use Waiting / Processing / Complete / Failed as primary states; explain scheduler subsets under Advanced details.
- Show Not configured / Disabled / No audio when applicable, rather than implying every video has missing work.
- Label totals that include sampled video frames, and distinguish them from standalone media counts.
- Use one plain-language error with Retry preserving the query/filters. Offer a path from failures to the appropriate recovery action.

## Acceptance Criteria

- [x] Every stalled/disabled/failed state explains its scope and a useful next step; recoverable errors offer Retry without losing context.
- [x] Add red→green coverage for behavior changes; verify desktop and mobile UI.

## Notes

- Integrated into the combined UX release. Independent feature and App integration reviews completed; parent validation passed.
- Scope: Statistics, App request errors/retry, focused helpers, API metadata/live wiring and bounded failed-job recovery. No Settings/Lightbox/Feed/Duplicates/navigation component edits.

### State and capability contract

- Primary states are Waiting / Processing / Complete / Failed. Waiting counts pending jobs plus expired or absent leases; Processing counts unexpired leases. This is queue state, **not a worker heartbeat**. Ready/runnable, pending (including delayed retries), leases, missing-job counts, raw failure errors and build metadata are under Advanced details. Queue scheduling is unchanged.
- Embedding denominator includes standalone images **and sampled video frames**. Image annotation counts standalone images only. Video annotation/transcription count videos. The media total excludes sampled frames.
- `/api/stats` and `/api/live` carry the same immutable `capabilities` snapshot from runtime options. `WorkerEnabled` is wired from `mode.startsWorker()`. For a local worker, annotations use `AnnotationsDisabled` and annotator presence; transcription uses `VideoTranscriptsOn` and transcriber presence. States are `configured`, `disabled`, `unconfigured`, `unknown`; zero jobs do not determine capability.
- API-only instances report worker/annotations/transcription as `unknown`, even if local API options or backend handles exist: they cannot know a separate worker’s configuration. Bare `stats.Collect` likewise returns unknown. Live refreshes retain the supplied runtime metadata without probing or mutating annotation configuration.
- Configured means a backend is wired, not that its model files, remote service or worker are currently healthy. Annotation settings and worker logs remain the next steps for those failures.
- Completed empty transcript counts are scoped to the current model’s completed transcription jobs. Copy says “completed without recognized speech”; the database cannot distinguish no audio from audio with no recognized speech. No definite “No audio” claim is made. There is no transcription Settings control; guidance names the worker’s Parakeet bundle/runtime configuration.

### Bounded recovery contract

- The pre-existing unfiltered `POST /api/jobs/retry-failed` also resets unrelated failures, requeues completed work missing annotations, and creates missing jobs including transcription. **The new UI never calls it.** Its compatibility behavior is intentionally unchanged.
- `POST /api/jobs/retry-failed?kind=<embed_image|annotate_image|annotate_video|transcribe_video>` updates only existing `failed` rows for that exact kind and current model. It resets attempts/lease/error/backoff and moves them to pending. It does not touch completed, pending, leased, other-kind or other-model jobs; it never inserts/backfills jobs. Response retains `{retried, enqueued_missing: 0}`. Empty/unknown kinds return 400; known local disabled/unconfigured kinds return 409 before mutation.
- Buttons clearly apply to all existing failures of the named step/current model, not just the recent 10-row sample. API-only unknown permits this bounded operation with explicit guidance to confirm the separate worker supports the step. No missing-job enqueue action was added.
- Capability metadata and bounded retry ship together. If an older server omits capabilities, the UI hides retry controls rather than risking a server that ignores `kind` and executes legacy blanket recovery.
- Recent failures link to their step’s recovery guidance. Annotation cards link to existing Settings. Success and failure stay visible beside the action; stats refresh errors separately identify a stale snapshot and offer Retry.

### Request recovery and scope safety

- One plain-language message replaces duplicate/raw transport errors. Retry reruns the same request scope, retaining exact text/tags/tag mode/NSFW/similar target/uploaded image and random-library seed. Failed append retries preserve the offset and existing current-scope cards; replacements show neither stale success counts nor stale cards/empty-success messaging.
- Request token, abort and request-key checks remain intact. A Retry attempt is explicit; leaving a failed scope and returning is a fresh replacement, not an accidental append. UX07 collection-detail scope handling is preserved.

### Verification / red → green

- Red: Go metadata/runtime/bounded-retry tests failed on missing contract fields; live metadata regression failed before live-handler wiring. Frontend helper tests failed on missing helpers. Built-browser UX08 test failed waiting for absent Retry. A separate old-server recovery guard assertion failed before the conservative capability gate.
- Green: `mise run check:frontend` (0 errors/warnings) and `mise run test:full` passed. Full task includes `go test ./...`, 57 frontend unit tests, all script regression tests (including missing-dist validation), frontend delivery build/asset validation, and the full browser smoke suite.
- Dedicated `scripts/atelier_health_test.mjs` is dynamically imported **after** `validateAtelierBuild`, preserving the missing-dist fixture that lacks test modules. It injects network and HTTP failures, verifies exact retry payloads and offset-48 recovery in desktop/mobile layouts, disabled/unconfigured/unknown/configured copy, denominators, honest empty-transcript wording, failed-job recovery success/failure and stats-refresh recovery. UX03/04 request-key races and UX07 regressions also pass.
- Backend tests cover metadata with no jobs, local disabled/unconfigured/configured options and API-only unknown, live snapshots before/after DB changes, expired versus live leases/delayed pending jobs, empty transcript model/state filtering, and bounded retry isolation/rejection.
- Screenshots: `/tmp/ux08/health-1280.png`, `/tmp/ux08/health-390.png`; mobile inspected for overflow/readability. Dependencies installed locally in this worktree; shared `deps/llama.cpp` was not changed or built. Existing native deprecation warnings remain. One earlier smoke run hit the pre-existing visual rank/timing assertion; subsequent complete runs passed unchanged.
- Parent reconciled shared API/smoke changes without dropping Settings, Feed, Lightbox or scope-guard coverage. Full suite (66 frontend unit tests), Svelte check (0 errors/warnings), jobs/stats/live/app race tests and formatting checks passed.
- Review found malformed kind queries could fall through to legacy recovery; red tests reproduced unintended backfill, then strict query parsing fixed the fail-open path. A second red→green test makes absent current-server job kinds show zero Processing, while legacy lease counts stay unknown. Both fixes independently re-reviewed with no remaining blockers.
- [HTML report, UX-08](../../docs/reviews/2026-09-22-atelier-ux-audit.html#UX-08) contains embedded screenshot evidence.
