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

- [ ] Retired URLs return the chosen response and no legacy shell assets/link ship.
- [ ] Root and emitted assets work in a clean build; missing dist is detected before release/container delivery.
- [ ] API/media authentication, cookie bootstrapping, media ranges/MIME and path protections remain tested.
- [ ] Common mise tasks, CI and packaging run Atelier verification without depending on legacy tests.
- [ ] Feature dispositions are documented without turning retirement into an open-ended parity project.

## Notes

- Architecture-only source inventory verified `internal/webui/http.go`, `internal/webui/http_test.go`, `internal/webui/static/`, `frontend/src/components/Rail.svelte`, `frontend/vite.config.ts`, root `package.json`, `scripts/ui_smoke_test*.mjs`, `Containerfile.cuda`, `.github/workflows/ci.yml`, and `scripts/package_release.sh`.
- No legacy removal or container change performed during the report-only audit.
- [HTML audit retirement plan](../../docs/reviews/2026-09-22-atelier-ux-audit.html#legacy).
