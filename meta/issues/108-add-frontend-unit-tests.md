# Add a frontend unit test runner for pure logic

## Summary

The frontend has no unit test runner. `frontend/package.json` offers only `dev`, `build`, `preview`, and `check`; the only coverage is the Playwright smoke suites at the repo root. Pure logic with real bug surface is untested: the feedback classifier and decay in `lib/feed.ts`, `deriveTitle`/`matchScore`/`tagTone` in `lib/utils.ts`, the URL round-trip in `lib/stores.ts`, and `combineLibraryPins` in `App.svelte`.

## Requirements

- Add vitest to `frontend/` with a `test` script, `environment: node` for `lib/`.
- Move `combineLibraryPins`, `randomKey`, and `newLibrarySeed` out of `App.svelte` into `lib/library.ts` so they are testable.
- Write tests for `lib/feed.ts`, `lib/utils.ts`, `lib/stores.ts` read/write URL, and `lib/library.ts`.
- Run `npm test` in `frontend/` from `mise run test` and CI (issue 104).

## Acceptance Criteria

- `cd frontend && npm test` passes and covers the four modules above.
- Tests run in CI.

## Notes

- Found during the 2026-09-21 review.
