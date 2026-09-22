# Complete and deploy the remaining recommended UX sweep

## Summary

User approved the remaining recommended UX work and a deployment after completion. Keep changes topical and reviewable; ship together with the already committed legacy retirement and UX-07 Collections improvements.

## Scope / subissues

- [x] UX-01/02: [Safe duplicate consolidation](preserve-collections-during-manual-duplicate-cleanup.md) and [honest match/keeper controls](clarify-perceptual-duplicate-safety-and-keeper-choice.md).
- [x] UX-03/04: [Navigation/mobile chrome](simplify-atelier-navigation-and-mobile-chrome.md) and [search/filter scope](clarify-search-filter-scope-and-match-labels.md).
- [x] UX-05: [Protect Settings drafts](protect-unsaved-atelier-settings-edits.md).
- [x] UX-06: [Lightbox viewing/editing actions](simplify-lightbox-actions-and-edit-mode.md).
- [x] UX-08: [Actionable status and error recovery](clarify-library-health-and-error-recovery.md).
- [x] UX-10: [Feed return and media states](make-feed-exit-destination-and-loading-state-explicit.md).
- [x] Deploy tested combined release to NAS CT107 and verify it.

## Requirements

- TDD, independent review before topical commits, formatting and full checks before commits.
- Preserve production data, settings, auth and media semantics. All editing/destructive browser tests use local fixtures only.
- Deployment: verified CT107 target, matching built Atelier assets, compatible Vulkan libraries, consistent SQLite snapshot and binary/link rollback. No Git push unless requested.
- Upload polish and unrelated performance/runtime followups are not included.

## Acceptance Criteria

- Scoped subissues completed with regression coverage and desktop/mobile checks.
- Production serves the final reviewed build and matching assets; existing data/settings preserved and worker/service health verified.
- Record deployment/rollback details; archive completed issues without deleting their notes.

## Progress

- `65708ee`: mobile/desktop navigation, explicit search scope and stale-result guards; independent review + full checks passed. UX04 final Lightbox score wording is coordinated with UX06.
- `de1f264`: memory-only Settings draft retention, explicit save/discard and unload warning; independent review + integrated checks passed.
- `6be6b6d`: transactional manual duplicate consolidation, metadata conflict checks, selectable keeper and honest visual-hash comparison; independent review + integrated checks/race tests passed.
- Inherited concurrent NSFW classification/display races and shared-dialog announcement improvements remain separately tracked; no unbounded follow-up expansion.
- UX10 integrated with mobile More navigation; full Go/frontend/script/browser suites and Svelte check passed, with independent integration review. Physical-device gestures remain outside this change.
- `9214642` plus UX08 UI: honest capability/status metadata, strictly bounded failed-job recovery and scope-preserving request Retry; independent feature/integration reviews, full suite/check and focused race tests passed.
- Final release seam review found no blockers. Optional integrated local MP4 playback check passed in Chromium; physical-device playback/gestures remain unverified. No database migrations, native dependency pin or Go dependency changes relative to outgoing `cc0aab4`.

## Deployment outcome — 2026-09-22

- Deployed `6b0aeead08e302e0c76a2a788f24ca296811db12` as `/opt/imgsearch/releases/6b0aeea-vulkan` on NAS CT107, replacing `cc0aab4-vulkan`. This includes legacy retirement, UX07 Collections and all scoped sweep work above. No Git push.
- Frozen clean source, incremental bundle and freshly built Atelier assets; exact full file/SHA256 manifests matched in CT107. Local full Go/frontend/script/browser suites, 66 frontend unit tests, Svelte check (0 errors/warnings), focused race tests and independent feature/integration/deployment reviews passed. CT107's stamped Go build and full Go suite passed with unchanged native sources and Vulkan libraries.
- First cutover (`20260922T144655Z`) failed: private root staging permissions made the candidate's sqlite-vector path inaccessible to the service user. Automatic binary/link rollback restored `cc0aab4`; no DB restore was performed. Post-rollback comparison confirmed schema/settings/config/counts unchanged. Both attempts' artifacts remain retained.
- Permission-only repair granted the service's actual primary group read/traverse access to the new release, removed group/other write access and left root ownership intact. It did not follow symlinks into older trees or change binary/frontend bytes. Added and independently reviewed pre-stop service-user candidate-version and in-memory SQLite-extension load checks; both passed. The runbook now records this required gate.
- Successful cutover at `20260922T145422Z`: stopped service, created an integrity-checked SQLite snapshot, retained old binary/link, installed candidate and restarted. API reports the exact target SHA. Service active with zero restarts after this cutover; Vulkan reports KHR_coopmat and 37/37 embedding layers offloaded.
- Preserved schema16, configuration/settings hashes, settings version and annotation settings. Counts unchanged from the immediate baseline: 4934 image rows (1540 standalone +3394 sampled frames),468 videos,0 collections/memberships. Remote annotation backend remains configured at concurrency2; all six output budgets unchanged. No failed/pending/leased jobs at verification; no imports, reannotations, recovery mutations or strict-dedup opt-in enabled for QA.
- Public root, JS and CSS returned200 and matched local/release SHA256. JS `index-DRsIqauK.js`: `7ac584a220cd61276a433e299916718c9f866670c116039431879416fd2a4be1`; CSS `index-Cn76FpM3.css`: `545fdd2185f913351f91cc9df06212c1c4621f6447a4afcfb322818f4c7edfb4`. `/healthz`200, `/legacy`404, anonymous Settings/Collections APIs401.
- Read-only production browser checks verified desktop/mobile Statistics/navigation, Settings with disabled Save changes and visible token controls, and empty Collections. Authenticated HTTP and WebSocket snapshots agree: worker local, annotation configured, transcription disabled. Transcription shows0 Waiting/Processing rather than suggesting468 queued jobs. Screenshots retained locally; no private media/audit screenshots published.

### Rollback

- Binary: `/opt/imgsearch/bin/imgsearch.backup-20260922T145422Z`.
- Previous-release link: `/opt/imgsearch/releases/current-before-6b0aeea-vulkan-20260922T145422Z` → `/opt/imgsearch/releases/cc0aab4-vulkan`.
- SQLite snapshot (root0600): `/var/lib/imgsearch/data/imgsearch.sqlite.before-6b0aeea-vulkan-20260922T145422Z`.
- Normal rollback stops service, restores the prior binary/link, then starts it; preserve the current database instead of discarding later writes. Retain historical native-library trees, including `1f8a123-vulkan`, referenced by copied RUNPATHs.

Acceptance criteria met; archived. Physical-phone playback/gestures, actual CUDA image build and the separately tracked UI follow-ups remain unverified/outside this batch.
