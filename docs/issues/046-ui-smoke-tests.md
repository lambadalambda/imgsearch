# 046: Wire Browser-Level UI Smoke Tests

## Priority

P2

## Status

Completed.

## Summary

The static UI has limited behavior-level tests. Add browser smoke coverage before attempting larger frontend refactors.

## Context

- Existing web UI tests mostly validate static asset content strings.
- `package.json` has Playwright helpers but no test runner.
- Core browser flows include initial render, tab navigation, upload modal behavior, tag search, NSFW refresh, and video pagination.

## Risks

- UI regressions in uploads, live updates, pagination, tag filters, modals, and keyboard behavior are easy to miss.
- Future frontend refactors will be riskier without a browser-level safety net.

## Acceptance Criteria

- [x] Add a Playwright smoke test command and wire it through `npm test` or a `mise` task.
- [x] Cover initial render, tab navigation, upload modal open/close, tag search basics, NSFW refresh, and video pagination.
- [x] Add keyboard handling for ARIA tabs or adjust markup to match implemented behavior.
- [x] Decide whether browser tests are included in `mise run test` or documented as a separate full-test task.

## Resolution

- Added `scripts/ui_smoke_test.mjs`, a Node/Playwright smoke harness with mocked API responses.
- Wired browser smoke tests through `npm test`, `npm run test:ui`, and `mise run test:ui`.
- Kept browser tests separate from `mise run test`, which remains the fast Go suite, and documented the setup in `docs/development.md`.
- Added keyboard navigation for ARIA tabs with disabled-tab skipping.

## Related Files

- `internal/webui/static/app.js`
- `internal/webui/static/index.html`
- `internal/webui/static/styles.css`
- `internal/webui/http_test.go`
- `package.json`
- `mise.toml`
