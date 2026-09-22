# Deploy configurable annotation token settings

## Summary

Deploy reviewed release `cc0aab4` to the existing home imgsearch service (NAS CT107), enabling Atelier's configurable generation limits.

## Requirements

- Verify authoritative NAS CT107 target, running version and existing configuration before cutover.
- Build/test matching frontend and Go release, preserving native Vulkan library compatibility.
- Retain rollback binary/release and a consistent database snapshot; preserve all data and settings.
- Do not enable opt-in strict import deduplication, change token values, requeue jobs, remove legacy UI, or push Git remotely as part of deployment.

## Acceptance Criteria

- Service is healthy on `cc0aab4`, authenticated Settings exposes all six budgets and the new UI assets are served.
- Existing configuration/data preserved; worker resumes without new infrastructure failures.
- Record verification and rollback paths; archive issue after successful rollout.

## Notes

- Operational task: implementation TDD does not apply. Use build/test gates and read-only deployment smoke checks.
- Follow the existing CT107 deployment runbook; never target CT100.

## Outcome — 2026-09-22

- Deployed `cc0aab4547d3874540362fa00bd201f8210f989b` as `/opt/imgsearch/releases/cc0aab4-vulkan`, replacing `907f8a9-vulkan`. Used the verified NAS CT107 target and existing unchanged CT-built Vulkan libraries; no push or configuration changes.
- Fresh local frontend build/check and full tests passed; CT107 Go build and focused DB/upload/annotation/worker/app/cmd/settings/native/remote tests passed. Full filename/SHA256 dist manifests matched before cutover; macOS AppleDouble archive metadata was removed from the unactivated release and binary rebuilt before deployment. Independent script review accepted the verified external artifact/host/build gates.
- Stopped service, integrity-checked a SQLite backup, retained prior binary/link, switched release and restarted. Rollback stamp `20260922T124542Z`: binary `/opt/imgsearch/bin/imgsearch.backup-20260922T124542Z`, link `/opt/imgsearch/releases/current-before-cc0aab4-vulkan-20260922T124542Z`, DB `/var/lib/imgsearch/data/imgsearch.sqlite.before-cc0aab4-vulkan-20260922T124542Z`. Normal rollback restores binary/link only; preserve current data rather than overwriting later writes from the snapshot. Retain older native-library release trees.
- Authenticated API reports target commit and all six default budgets. Stored settings/config hashes unchanged; backend remains remote, concurrency2. Counts unchanged:4935 images,468 videos,0 collections. No migration (schema16), imports, reannotation or strict-dedup opt-in enabled.
- Public root/assets200; JS `index-Bzclga0Y.js` SHA256 `8b56fb28544a2215b9a0fa4babeff024e4010815d8f38c7eb06853e7eba2cf12` matches local/release. Anonymous settings401. Service active with zero restarts, queue4935/4935 done, failed0; Vulkan retains KHR_coopmat and37/37 embedding layers offloaded.
- Acceptance criteria met; archived. Legacy retirement and UX-07 will be separate subsequent changes, not part of this deployed release.
