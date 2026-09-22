# 063: Filter Upload Drop Files Client-Side

## Priority

P3

## Status

Open.

## Summary

The Upload modal's file picker uses accepted MIME/extensions, but drag-and-drop files bypass that filter and unsupported files can enter the pending list.

## Context

- `frontend/src/lib/api.ts` defines `UPLOAD_ACCEPT`.
- The Upload file input uses `accept={UPLOAD_ACCEPT}`.
- `frontend/src/components/Upload.svelte` accepts every file from `DragEvent.dataTransfer.files` in `addFiles`.
- Unsupported files only fail after submitting to the server.

## Risks

- Users can queue unsupported files and only learn after a failed upload round-trip.
- Pending rows can imply unsupported files are valid.
- Client/server validation behavior is inconsistent.

## Acceptance Criteria

- [ ] Add smoke or unit coverage for dropping an unsupported file.
- [ ] Filter dropped and picked files through a shared accept predicate.
- [ ] Show a clear rejected-file message without adding unsupported files to pending rows.
- [ ] Keep server-side validation authoritative.

## Related Files

- `frontend/src/components/Upload.svelte`
- `frontend/src/lib/api.ts`
- `scripts/ui_smoke_test_atelier.mjs`

## UI/UX audit 2026-09-22

Browser file-input automation supplied a .txt file alongside a valid WebM. Both entered the pending queue. The server then rejected the .txt and accepted the video, with clear per-file outcomes. This bypassed the native chooser's accept filter; it is not proof the normal chooser lists unsupported files.

Additional recommendation: Offer Retry failed for transient failures and a correction/remove action for unsupported formats; preserve successful rows.

See [HTML report, UX-09](../../docs/reviews/2026-09-22-atelier-ux-audit.html#UX-09).
