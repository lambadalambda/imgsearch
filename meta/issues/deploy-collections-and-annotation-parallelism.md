# Deploy collections and annotation parallelism to home instance

## Summary

Deploy tested application commit `907f8a9` to `https://imgsearch.home.heldscal.la/`, including saved media collections, actual remote annotation concurrency, and the standalone Fediverse import tooling.

## Requirements

- Verify authoritative `root@nas` / CT107 target and running revision before changes.
- Build a clean commit-specific Vulkan release using the established runbook; preserve configuration/models/media and existing settings.
- Back up SQLite consistently before migration 16 and retain prior binary/current-link rollback artifacts.
- Minimize service interruption; do not trigger imports or bulk reannotation to test deployment.

## Acceptance Criteria

- Service/API report target commit; public frontend assets match the tested local build.
- Collection API works with authentication and migration 16 is applied without losing existing media.
- Current remote parallel-annotation setting is retained; worker/service health checked after restart. Observe overlap if queued work exists, without manufacturing paid inference jobs.
- Deployment and rollback details recorded; no secrets printed/stored.

## Notes

- Operational deployment: TDD does not apply; use reviewed code, build/test gates, and deployment smoke checks.
- Source: Obsidian `opencode/workflows/imgsearch-lxc-deployment.md`. Push not requested; ship a git bundle directly as in prior deployments.

## Outcome — 2026-09-22

- Verified CT107 (`llamacpp-vulkan`) was serving `5324a80`; built clean `/opt/imgsearch/releases/907f8a9-vulkan` from an incremental bundle and locally built Atelier assets. No GitHub push or configuration/model changes.
- Local frontend check/build and full Go/frontend/script/browser suite passed. CT107 Go 1.24.2 build and focused database, collections, worker, app, command, settings, and remote-annotator tests passed. Reused unchanged CT-built Vulkan libraries; all older native-library releases retained.
- Independently reviewed the cutover/rollback script. Stopped service, created and integrity-checked a SQLite backup, preserved old binary/link, installed target, and restarted successfully. Migration 16 applied from 15; image/video counts stayed at 4,961 / 468.
- Rollback stamp: `20260922T085335Z`. Binary: `/opt/imgsearch/bin/imgsearch.backup-20260922T085335Z`. Prior link: `/opt/imgsearch/releases/current-before-907f8a9-vulkan-20260922T085335Z` → `5324a80-vulkan`. Database: `/var/lib/imgsearch/data/imgsearch.sqlite.before-907f8a9-vulkan-20260922T085335Z`. Normal rollback should restore binary/link while retaining the additive schema and current data; do not overwrite subsequent writes with the snapshot automatically.
- Live API reports `rolling-20260922-907f8a9`, full commit `907f8a9c433dcefd407bdbb76df3d5e50782bbd7`. Public root/JS returned 200; asset `index-CDglaCYr.js` SHA256 matched local/release (`26c25597b2673b7f428d0f62420b44bce81bf08d7430bfac79c14f20151de6cc`).
- Authenticated temporary collection create/rename/mixed image+video save/idempotence/delete passed. Removed only that temporary list; original media remained accessible. Anonymous collection API returned 401.
- Remote backend and saved concurrency 2 preserved. Observed two annotation leases simultaneously and interleaved job claims/completions; queued work continued with zero failed jobs and zero worker infrastructure errors. Vulkan retained `KHR_coopmat` and 37/37 layers offloaded. No test imports or reannotation jobs created.
- Acceptance criteria met; archived. Obsidian deployment runbook updated with release/rollback details (no credentials).
