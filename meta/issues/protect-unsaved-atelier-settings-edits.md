# Protect unsaved Settings edits

## Priority

P2 — Reproduced workflow issue.

## Summary

A simple navigation click silently throws away the Settings draft.

## Evidence

In the sandbox, changed the native variant from e4b to 26b without saving, navigated to Library, then back to Settings: e4b returned and Save was disabled. Repeated via Collections. There was no discard warning or retained draft. No model configuration was saved.

Reproduction / inspection steps:

1. Settings → change a field; do not save.
2. Choose Library or Collections, then return to Settings.
3. The prior value is restored without warning.

## Risks

The same behavior can cost a longer remote-server configuration edit. Global search and tag links remain prominent while editing, making accidental departure easy.

## Requirements

- Add a dirty-state guard with Stay / Discard changes, or retain an explicit unsaved draft across internal navigation.
- Keep a Save changes / Discard bar near edited controls, and show a clear Saved confirmation.
- Related layout improvement: avoid equal-height Settings cards that create a large empty backend panel; use independent panels or a compact form layout.

## Acceptance Criteria

- [x] Leaving a dirty form cannot silently erase edits; the user can distinguish draft, saved, and active worker configuration.
- [x] Add red→green coverage for behavior changes; verify desktop and mobile UI.

## Notes

- Implemented in the UX05 worktree; awaiting parent review/merge. No index/archive/changelog changes or commit/deploy/push.
- `frontend/src/lib/settingsDraft.ts` owns an in-memory draft and save lifecycle, with pure form/request/dirty helpers. Settings remount refreshes the loaded baseline without replacing dirty inputs. Discard uses that latest loaded baseline; save success clears dirty state and the entered API key; save failure retains the complete draft. Duplicate saves and stale pre-save GET responses are guarded; inputs are disabled during saving.
- The page-lifetime draft arms `beforeunload` only while dirty, including when Settings is unmounted. Neither localStorage nor sessionStorage stores the draft or API key. Accepting a reload intentionally drops unsaved edits; dismissing the browser warning preserves them. Browser-native warning text and delivery are browser-controlled (not guaranteed on mobile process termination).
- Settings shows **Unsaved changes** / **Saved**, a nearby sticky **Save changes / Discard changes** bar, and a separate **Active worker configuration** line. Backend panels use independent heights. App/Rail/search/Feed and shared API types were not edited; native source/build and the existing llama.cpp symlink were not modified.

### Verification evidence

- **Red (unit):** `npm --prefix frontend test -- settingsDraft.test.ts` failed before the helper existed (`Cannot find module './settingsDraft'`).
- **Red (browser):** the new desktop/mobile smoke helper against the original UI failed `reload warning survives unmount` (false instead of true), reproducing the missing page-lifetime protection.
- **Additional red→green regression:** extending browser coverage to the stored-key checkbox exposed `Clicking the checkbox did not change its state`. Store feedback updates during the checkbox's `input` event preceded its `change` binding. Deferring that feedback path to `change` fixed it; the key-clearing navigation/discard assertion remains.
- **Green:** all **53 frontend unit tests in 8 files** passed, including four draft suites covering all form field kinds, key replacement/clearing, latest-baseline discard, retained draft on refresh failure, duplicate/in-flight saves, and stale load suppression.
- **Green:** `mise run check:frontend` reported **0 errors / 0 warnings**. `git diff --check` passed.
- **Green:** `mise run test:ui` rebuilt and validated embedded Atelier assets, then passed the full browser smoke suite. UX05 checks run at **1440×900 and 390×900**, covering Library/Collections/browser-back/Feed round trips; native and remote fields; save/discard; pending save unmount/remount success and failure; actual reload dialog dismissal outside Settings and acceptance inside Settings; clean-state guard removal; distinct saved versus active worker values; compact panel heights and viewport fit; and no API key written to browser storage or restored after reload.
- An earlier full smoke run hit an unrelated existing visual search-rank ordering assertion. A rerun and the final `mise run test:ui` both passed; no search changes were made.
- JavaScript dependencies were installed from existing lockfiles, without dependency changes. The existing frontend audit reports five vulnerabilities (four high, one moderate); dependency upgrades are outside UX05. Go/native tests were not run for this frontend-only change. Independent parent review remains pending.

- [HTML report, UX-05](../../docs/reviews/2026-09-22-atelier-ux-audit.html#UX-05) contains embedded screenshot evidence.

## Completion

- Independent review found no draft-retention/persistence blockers. Integrated with new mobile More navigation; deferred test-module import preserves missing-dist fail-fast regression. Parent full suite/check passed (57 frontend tests plus all Go/script/browser checks). Archived.
- Inherited stale connection-probe feedback and simplified credential fixture are recorded in the existing token-draft/probe validation issue, not silently expanded into this change.
