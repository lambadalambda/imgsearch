# Review repository and recommend next priorities

## Summary

Assess the current implementation and open backlog to recommend a short, evidence-based next-work sequence. No application changes are in scope.

## Requirements

- Inspect current code, recent changes, and existing issues rather than treating the backlog as authoritative.
- Separate confirmed gaps from unverified risks and avoid duplicate follow-up issues.

## Acceptance Criteria

- Deliver ranked recommendations with source references and practical completion criteria.
- Report verification performed and its limitations.

## Notes

- This is an assessment; behavioral TDD does not apply.

## Findings and recommended sequence

1. **Persisted image thumbnails (079):** `frontend/src/lib/utils.ts:61–68` uses the original for both thumbnail and full media. Reuse the existing database thumbnail column/list field/deletion handling, generate bounded derivatives, expose them in search results too, and backfill old libraries. Verify grid requests use derivatives while the lightbox uses originals. Highest broad user-facing payoff; actual bandwidth savings still need measurement.
2. **Tag correctness, then scale (066, 092):** fix missing video-level tags in the cloud first, independently of a schema change. The cloud expands only image/frame JSON (`internal/search/http.go:1385–1415`), whereas annotator vocabulary also unions video JSON (`internal/db/tags.go:20–41`). Then benchmark/materialize shared tag data. Keep conservative aliases and preservation of manual additions/removals in a separate 092 change; vocabulary prompting is already implemented but does not canonicalize stored tags.
3. **Verified rolling releases:** `.github/workflows/rolling-release.yml` publishes after packaging only, independently of CI. Require verification of the same SHA before publication, including manual dispatch. See the dedicated follow-up issue.
4. **Small Feed/upload reliability fixes (062, 064, 063):** visible empty/error states instead of silent Feed launch failures; exclude control-origin touches from swipes; filter unsupported dropped files before queueing. Existing browser smoke coverage is green, not proof that these untested paths work.
5. **Finish quality-gated performance evaluation (070–073), not speculative tuning:** the harness already exists (069). Recorded results in `docs/ingestion-benchmarks.md:92–98` show 165s compact-frame baseline versus 171s n-gram speculation, and a rejected faster prompt that reduced NSFW coverage from 10 images to 0. Finish representative quality comparisons and repeated timing runs before changing resolution, projectors, or output policies. These are historical measurements, not new benchmarks from this review.

## Backlog reconciliation

- Verify remaining acceptance criteria for the open mobile Feed/vector-feedback umbrella issues: substantial implementation and tests already exist.
- Reconcile 069 against the implemented harness and documented CPU/CUDA runs rather than building it again.
- Narrow 073 to remaining scene-sampling/quality work; compact frame evidence and exact-frame evidence reuse already shipped.
- Keep mixed-model search hardening (050), native lock tuning (040), and batched native inference (014) below these priorities absent new profiling evidence or multi-model requirements.
- No existing issue was closed solely on source inspection during this review.

## Verification

- `mise run build:frontend`: passed.
- `mise run check:frontend`: zero errors/warnings.
- `mise exec -- sh -c 'gofmt -l ./cmd ./internal'`: no unformatted Go files.
- `mise run test:full`: passed (Go suite, 25 frontend unit tests, script tests, legacy and Atelier browser smoke suites).
- Go results were largely cached. Opt-in native inference integrations, GPU/CPU benchmarks, large-library profiling, and live manual browser QA were not run.
- Independent read-only backend and frontend assessments informed the recommendations. No application code changed.
