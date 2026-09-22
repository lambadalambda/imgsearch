# Deploy image thumbnails to the home instance

## Summary

Deploy tested application commit `5324a80` to `imgsearch.home.heldscal.la` using the documented Obsidian LXC workflow.

## Requirements

- Verify the authoritative target and running version before changes.
- Build in a commit-specific release directory; preserve service configuration, models, and library data.
- Follow the existing rollback/backup procedure and minimize service interruption.

## Acceptance Criteria

- The running service reports the intended commit and the public UI serves matching frontend assets.
- An authenticated image thumbnail request returns a bounded WebP; the original remains available.
- Service health and queue state are checked after restart.

## Notes

- Operational deployment; no new behavior to implement with TDD. Use the tested code and deployment smoke checks.
- Source instructions: Obsidian `opencode/workflows/imgsearch-lxc-deployment.md`.

## Outcome — 2026-09-22

- Verified `root@nas` / CT107 (`llamacpp-vulkan`) was running `ee63457`; prepared `/opt/imgsearch/releases/5324a80-vulkan` using an incremental git bundle and locally built Atelier assets. No GitHub push was needed.
- Reused the unchanged native submodule's CT-built Vulkan libraries; built the stamped binary inside CT107 with Go 1.24.2/libvips 8.15.1. Thumbnail, imageutil, images, search, and app tests passed there.
- Independently reviewed the switch/rollback script, then stopped the service, backed up SQLite, preserved the binary/release link, installed the new binary, and started the service. No configuration/model/schema changes.
- Rollback artifacts: `/opt/imgsearch/bin/imgsearch.backup-20260922T064500Z`, `/opt/imgsearch/releases/current-before-5324a80-vulkan-20260922T064500Z` (points to `ee63457-vulkan`), and `/var/lib/imgsearch/data/imgsearch.sqlite.before-5324a80-vulkan-20260922T064500Z`. Keep older release trees because copied native libraries retain existing RUNPATH dependencies.
- Verified live build `rolling-20260922-5324a80` / full commit `5324a80cb06eb7fc6a0d039a3fcb891d92b0761b`; public root and JS asset returned 200 and the asset checksum matched the release.
- A real public authenticated thumbnail returned 480×640 WebP / 30,686 bytes, while its 173,440-byte original remained available. Anonymous thumbnail returned 401; authenticated conditional request returned 304.
- Service active; Vulkan reported `KHR_coopmat` and 37/37 layers offloaded. Queue remained 4,061 done, zero failed/pending/missing. Acceptance criteria met; deployment archived.
