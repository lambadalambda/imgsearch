# Configure annotation output token budgets in Settings

## Summary

Allow users to tune generation limits for standalone images, sampled video frames and final video summaries without recompiling or restarting.

## Requirements

- Persist six output-token budgets: first attempt and compact retry for each annotation kind.
- Preserve current defaults for existing installations and clients omitting the new settings.
- Validate bounded positive integer values; expose controls in Atelier Settings for both native and remote backends.
- Apply saved limits to subsequent annotation requests, including native cached runtimes, without changing in-flight requests.
- Explain that these are output limits, not image preprocessing or context-window sizes; increasing them does not guarantee successful generation.

## Acceptance Criteria

- Regression tests cover defaults, persistence, validation, API compatibility and runtime request propagation including retries.
- Settings controls load/save all six budgets and pass frontend checks and UI smoke tests.
- Full tests, formatting and independent correctness/architecture review pass before commit.

## Notes

- Scope excludes deployment, automatic retries of existing failed jobs, legacy UI removal and finish-reason diagnostics.

## Completion

- Added persisted `annotation.output_tokens` values with original defaults and integer bounds `64..32768`; old settings and omitted/partial updates remain compatible.
- Both native implementations and remote requests receive immutable primary/retry budget pairs from the resolver. Budget-only saves do not rebuild backends; active requests retain their original limits.
- Added shared Atelier Settings controls for images, sampled frames and video summaries, each with first-attempt and compact-retry inputs.
- TDD: backend tests first exposed missing settings/propagation and fixed request budgets; UI smoke first failed on the absent output-limits section. All passed after implementation.
- Verification: fresh frontend build, `mise run test:full`, `mise run check:frontend`, focused Go race tests, formatting and `git diff --check` passed. Native CGO compiled and generator-seam tests passed; model-backed inference was not exercised.
- Independent backend and frontend reviews found no blockers. Tracked separate followups for native output capacity, explicit-null API semantics and draft validation during remote probes rather than broadening this change.
- Updated architecture notes and changelog. No deployment or requeue of existing failed jobs.
