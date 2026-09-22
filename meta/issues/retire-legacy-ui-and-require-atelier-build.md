# Retire the legacy UI and require a working Atelier build

## Priority

P2 — approved product direction; implementation followup.

## Summary

The user explicitly permits removing the legacy UI entirely. Make Atelier the only supported shell without changing shared API/media/auth behavior or accidentally shipping a build with no functional frontend.

## Requirements

- Remove legacy navigation, `/legacy` routes/assets, static shell files, and the missing-Atelier fallback's legacy link. Prefer 404 for retired URLs; redirect only if bookmark compatibility is deliberately wanted.
- Remove legacy-only smoke tests and npm scripts; retain Atelier browser testing, Go frontend/media tests and Playwright.
- Preserve authentication, root browser-cookie bootstrapping, `/assets`, `/favicon.ico`, API endpoints, authenticated media MIME/range responses, and directory/path protection.
- Require built Atelier assets in all supported delivery paths. Source inventory found that the CUDA Containerfile compiles Go without building Atelier and relies on prebuilt dist; the Atelier smoke test skips successfully when dist is absent. Retirement must not turn those into a successful build of a placeholder-only app.
- Update current documentation and task references while preserving historical decisions/issues.
- Do not automatically port every legacy-only feature: explicitly choose what to retain, replace, or retire. Inventory found bulk image deletion, retry-stuck-jobs UI, and dedicated tags/live-connection UI as decisions; backend APIs can remain.

## Acceptance Criteria

- [x] Retired URLs return the chosen response and no legacy shell assets/link ship.
- [x] Root and emitted assets work in a clean build; missing dist is detected before release/container delivery.
- [x] API/media authentication, cookie bootstrapping, media ranges/MIME and path protections remain tested.
- [x] Common mise tasks, CI and packaging run Atelier verification without depending on legacy tests.
- [x] Feature dispositions are documented without turning retirement into an open-ended parity project.

## Notes

- Architecture-only source inventory verified `internal/webui/http.go`, `internal/webui/http_test.go`, `internal/webui/static/`, `frontend/src/components/Rail.svelte`, `frontend/vite.config.ts`, root `package.json`, `scripts/ui_smoke_test*.mjs`, `Containerfile.cuda`, `.github/workflows/ci.yml`, and `scripts/package_release.sh`.
- No legacy removal or container change performed during the report-only audit.
- [HTML audit retirement plan](../../docs/reviews/2026-09-22-atelier-ux-audit.html#legacy).

## Implementation and handoff

- Scope: retire the shell with 404s, keep shared backend/auth/media behavior, and gate supported delivery on built Atelier assets. No deploy/push or commit; parent owns independent review, full-check handoff, and index archival.
- Disposition: retire legacy-only bulk-delete UI, dedicated tags explorer and live-connection panel. Keep backend APIs. Retry-failures UX stays in the existing UX-08 issue, not automatic parity work here.
- Validation plan: red→green retired-route/fallback tests, real-root cookie/media integration, missing/placeholder/incomplete dist regressions (including smoke and skip-build packaging), then fresh frontend check/build, local Go/frontend/script/browser suites and formatting.


### Source changes

- Removed `internal/webui/static/{index.html,app.js,styles.css}`, the embed and legacy index handler, `/legacy` registrations, Rail escape link, Vite proxy, and fallback link. The exact root still serves Atelier/build-help HTML; unknown paths (including retired paths on GET/HEAD/POST) return 404. `/assets/`, `/favicon.ico` and media handlers remain.
- Removed `scripts/ui_smoke_test.mjs` and `test:legacy`; retained Playwright and all Atelier/shared Go tests. Root npm test scripts run Atelier only. Browser smoke also asserts no retired navigation.
- Added `scripts/validate_atelier_build.mjs` and `scripts/atelier_build_test.{mjs,sh}`. The shared validator rejects missing index, placeholder/source index, missing referenced files and empty assets; it requires the app mount plus emitted module JS/CSS. It is deliberately a structural check, not a browser emulator or dependency graph parser.
- Frontend `npm run build` validates after Vite; release packaging validates unconditionally before Go even with `IMGSEARCH_SKIP_FRONTEND_BUILD=1`; startup smoke validates before building Go; browser smoke validates before importing Playwright. Loading Playwright only after the gate lets script regressions run in CI before root npm dependencies are installed.
- CUDA container now compiles/validates Atelier in a separate Node 24 stage and copies only that output into the Go stage. `.dockerignore` excludes host dist and node_modules. CI frontend build/check precedes Go tests; `mise run test:ui` builds first and `test:scripts` includes build-gate regressions.
- Updated active README/development/frontend/architecture/mockup instructions, AGENTS.md and source comments that pointed to deleted files. Added ADR-012 without rewriting ADR-005/008, prior changelog entries, archived issues or the audit report. No UX-07 or Collections implementation changed.

### Explicit feature disposition

- Retire legacy-only bulk-select/delete-all UI, the dedicated tags explorer and live-connection panel/badge. Keep Atelier per-item deletion, tags/chip search, live counts and media-status behavior as-is.
- Retain shared backend bulk-delete, tags, live and job-management APIs. No schema, data or backend capability removal.
- Retry-failures improvements remain [UX-08: Turn operational states into next steps](clarify-library-health-and-error-recovery.md); retirement does not promise or implement an ops-menu port.

### Red → green evidence

1. Before implementation, `go test ./internal/webui ./internal/httputil` failed on retired routes (GET/HEAD returned 200, POST returned 200/405), the fallback's legacy link, and `/legacy` under the real cookie/auth middleware chain. Shared middleware tests stayed green. The new integration test also verifies an anonymous root produces the browser cookie, API/media requests without credentials remain 401 without minting cookies, and that cookie authorizes the API and a `206 video/mp4` range response.
2. Before implementation, `node --test scripts/atelier_build_test.mjs` failed: smoke skipped missing dist with status 0; skip-build packaging reached the fake Go compiler without assets; validator was absent; CUDA/delivery wiring still lacked a frontend build.
3. A follow-up red regression reproduced clean CI's missing-Playwright dependency when running the missing-dist test outside the installed root. Moving Playwright import after asset validation made it green without requiring browser dependencies for script tests.
4. After implementation, all 10 build-gate regressions pass. Go webui/auth regressions pass with the fresh embedded build, including actual emitted JS/CSS responses, favicon routing, retained MIME/range and directory/path checks. The source-only fallback remains tested via an empty in-memory filesystem.

### Local verification

- `mise run check:frontend`: **pass**, 0 errors / 0 warnings.
- `mise run build:frontend`: **pass**, clean-dist + fresh Vite build + validator; emitted JS/CSS present.
- `mise run test:full`: **pass** after the fresh build: all Go packages, 44 frontend unit tests, every configured shell/script suite (including the 10 gate regressions), another fresh Atelier build and Playwright browser smoke.
- `go vet ./...` and `go test -race ./...`: **pass**.
- `gofmt` applied to changed Go files; `gofmt -l ./cmd ./internal` empty. `git diff --check`, changed shell `bash -n` and JavaScript `node --check`: **pass**.
- Local ephemeral logs: `/tmp/legacy-red-go.log`, `/tmp/legacy-red-build.log`, `/tmp/legacy-ci-red.log`, `/tmp/legacy-full-final.log`, `/tmp/legacy-race.log` (not repository artifacts).

### Cautions / remaining handoff

- **No container image build:** local Podman exists but cannot connect to its socket (`127.0.0.1:56766`, connection refused). Container stage/order/context are regression-checked; an actual image build still needs a working container host. No system dependencies installed or VM started.
- No full native release archive was produced; packaging's missing-dist/skip-build path is executed in an isolated fixture with a fake Go compiler, and existing release configuration tests pass.
- The developer empty-dist fallback intentionally remains: bare `go build` supports backend-only development, not release delivery. Direct browser smoke now requires a real build; `mise run test:ui` builds automatically. Node is required for release asset validation even when reusing dist.
- The validator proves index/asset structure and nonempty referenced files, not arbitrary JS correctness; retain browser smoke. Current frontend has no favicon asset, so its preserved endpoint correctly returns 404 rather than the HTML shell.
- Production remains at the already-deployed `cc0aab4`; no deployment, push or commit performed. Acceptance implementation is ready for the parent's independent review/full-check/commit handoff. Leave `meta/issues.md` and `meta/issues_archive.md` untouched until the parent archives it.

## Completion

- Independent correctness/architecture review found no blockers. Clarified the mise-versus-direct-npm browser-test instructions. Parent reran `mise run test:full`, `mise run check:frontend`, formatting and whitespace checks successfully.
- Archived after verified local implementation. No subsequent deployment or push; production remains `cc0aab4`. Container image-build limitation above remains explicit.
