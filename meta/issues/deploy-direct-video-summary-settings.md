# Deploy direct video summary settings

## Summary

Deploy source `a7fd679f2c3158188b62d8322049513d585bbf19` to NAS CT107 (`llamacpp-vulkan`), replacing `6b0aeea`.

## Requirements

- Preserve saved settings and media; do not enable direct mode, change sample count, import, resample, or reannotate.
- Match tested source and the complete frontend asset manifest; reuse the unchanged native dependency build.
- Validate candidate permissions, binary revision, and sqlite-vector loading as the service user before stopping production.
- Keep binary/link rollback and an integrity-checked SQLite snapshot; never restore the database automatically.

## Acceptance Criteria

- Fresh full local checks and CT107 Go tests pass; independent deployment-gate review completed.
- Public assets and authenticated served revision match the candidate; service remains healthy.
- Raw settings/configuration, schema, and media counts are unchanged; added API fields normalize to existing defaults.
- Read-only production Settings inspection confirms the controls are live without saving changes.
- Record release and rollback paths and archive this issue.

## Notes

Operational verification replaces implementation TDD for this deployment. No migration, native pin, or Go dependency changes. Preserve historical release trees used by library RUNPATHs. Candidate release permissions must use the actual service primary group, not an assumed `imgsearch` group.

## Outcome

Deployed successfully on 2026-09-22 at 16:16Z. Active release: `/opt/imgsearch/releases/a7fd679-vulkan`; authenticated served revision matches the full source SHA above. No push or settings/media mutations were performed.

- Fresh `mise run test:full`, `mise run check:frontend` (zero errors/warnings), and `git diff --check` passed. CT107 full Go suite, exact frontend manifest, binary checksum, dependency resolution, and service-user binary/SQLite probes passed.
- Independent deployment-gate review found no blockers. Automatic rollback protects startup/served revision; subsequent read-only settings/public acceptance gates also passed.
- Public root/JS/CSS hashes match the tested build. JS `index-eAyhZy-n.js`: SHA256 `2740b1c46a1242d3bf87145f2adca16ce58122dcd5ee16778b43a26c29b7bdad`. `/healthz`200, `/legacy`404, anonymous settings/collections APIs401.
- Service active/running, zero restarts; Vulkan KHR_coopmat, 37/37 layers offloaded. Remote backend remains at concurrency2; transcription remains disabled; zero pending/leased/failed jobs.
- Raw settings/config hashes, settings version, all existing annotation fields/token budgets, schema16 and counts unchanged:4934 image rows,468 videos,0 collections/items. New normalized values are `frame_descriptions` and5; the saved legacy JSON was not rewritten.
- Read-only live browser inspection confirms both mode options, enabled remote selector, count5 constrained to1–32, timestamp/provider/cost/new-upload-only guidance, and no horizontal overflow. No settings save or reannotation was invoked.
- Provider-specific multi-image acceptance, output quality and performance remain untested; direct mode was deliberately not enabled.

### Rollback artifacts

- Binary: `/opt/imgsearch/bin/imgsearch.backup-20260922T161605Z`
- Prior link: `/opt/imgsearch/releases/current-before-a7fd679-vulkan-20260922T161605Z` → `/opt/imgsearch/releases/6b0aeea-vulkan`
- Integrity-checked SQLite snapshot: `/var/lib/imgsearch/data/imgsearch.sqlite.before-a7fd679-vulkan-20260922T161605Z`

Normal rollback restores binary/link only, retaining the current database. Preserve historical native-library RUNPATH trees.
