# Manual duplicate consolidation

The Duplicates pane lists **Possible matches**, not verified copies. dHash compares a small grayscale visual signature; equal hashes do not establish byte or pixel equality. Colors, edits and fine detail can differ. Groups are connected components at the chosen distance, so not every pair in a group necessarily matches directly.

## Review and keeper choice

- **Identical visual hash** means distance zero. Strict/Normal/Loose retain distances 2/4/7.
- The initial keeper has the most pixels, with image ID breaking ties deterministically. This is not a quality ranking. Identical width and height say **Same resolution**; equal pixel counts with different dimensions say **Same pixel count**.
- Choose any group member with the Keep radio control. The comparison shows full, uncropped originals in two columns, including on phones; Flip comparison alternates the same pair. All controls are keyboard accessible.
- Format comes from the stored MIME type. File size is best effort: duplicate discovery stats regular files through `os.OpenRoot`, which confines access to the configured data directory even across symlinks. Missing, inaccessible or unsafe paths show **Size unavailable**.
- Review names the keeper, the number of other library entries removed, and the distinct collections containing any removed entry. Collections already containing the keeper are included in this count.

## Atomic database operation

`POST /api/duplicates` previews by default:

```json
{
  "keeper_id": 12,
  "group_ids": [12, 13, 14],
  "distance": 4,
  "include_nsfw": false
}
```

The response contains `keeper_id`, `removed_ids`, `affected_collections`, `file_policy: "retained"`, `conflicts` (an array of messages), and `preview_token`. Confirm by sending the same selection with `confirm: true` and that `preview_token`.

The service rejects malformed/duplicate/nonpositive IDs, a keeper outside the selection, unknown fields, invalid distances, and groups outside 2–500 members (400). It recomputes the complete currently visible standalone-image component using the same hash grouping and NSFW filter as discovery. Missing members, unrelated images, unhashable images, video frames, partial/stale groups, stale previews and unresolved conflicts fail closed (409). Hidden/unselected records are never implicitly included or deleted. The request accepts no filesystem paths.

Confirmation acquires SQLite's writer reservation before reading. Membership validation, metadata validation, metadata merge, collection union, dependent embedding/job deletion and image-row deletion share one transaction. Failure rolls everything back. Each collection keeps the keeper's existing membership/order ID when present; otherwise its earliest membership among the removed images is retargeted to the keeper. Other overlaps are removed.

The stateless preview token is a SHA-256 freshness checksum over the selection, content identity/hash/dimensions, relevant metadata and exact membership rows—not an authorization token. Existing API authentication remains responsible for authorization. Confirmation always recomputes and revalidates the operation; it does not trust client deletion IDs or counts.

## Manual metadata policy

- Preserve a single distinct nonempty manual title. Different nonempty manual titles block cleanup, rather than letting the keeper silently win.
- Union manual tag additions and manual tag removals separately, case-insensitively, using existing tag normalization/merge utilities. Ascending image ID determines stable spelling/order.
- If a tag is both manually added and manually removed anywhere in the group, block cleanup and name the conflicting tag. Invalid tag JSON also blocks cleanup.
- Conflicts are shown and focused before a confirmation dialog is offered. Edit the conflicting images using the existing metadata editor, then review again. Changing keeper is not a conflict-resolution override.
- Generated annotations, timestamps, dimensions, content and thumbnails remain those of the keeper. Served tags are recomputed from the keeper's generated tags and the merged manual additions/removals. This deliberately does not combine different generated descriptions or copy embeddings from different content.

The unattended strict-import operation is separate and unchanged. Its replacement-copy code assumes a fresh incoming row, so it is not a safe merge helper for two established library records. There is no unattended-policy status exposed in the current settings API; this pane does not invent one or change that setting.

## File retention and recovery

**Original files and thumbnails are retained. No files are deleted by consolidation, and no disk space is reclaimed.** Digest-based file paths can be reused by a concurrent import; postcommit unlinking could remove a newly imported file. Retention avoids that race and permits out-of-band recovery. Database removal cannot be undone in the UI; retained bytes alone are not a metadata backup. File garbage collection and a restore UI are intentionally outside this bounded change.

## Regression checks

- `mise run test`: includes transaction/rollback, collection ordering/union, manual metadata/conflicts, stale previews, invalid selections, standalone/transitive/NSFW boundaries and safe file-size lookup; strict-import tests remain unchanged. Frontend unit tests cover labels and preview/confirm API serialization/errors.
- `mise run check:frontend`: Svelte/TypeScript checking.
- `mise run test:ui`: validates the delivery build, runs the existing Atelier smoke suite and `scripts/duplicates_ui_test.mjs`. Focused regressions run at 1280×900 and 390×844: keyboard alternate keeper, tie/format/size labels, two-up layout without overflow, uncropped originals, flip controls, focused conflict reporting, cancel/no writes, retained-file/count preview, stale-confirm failure without optimistic removal, and successful confirmation.
- `mise run test:scripts`: existing delivery/script checks.

No new runtime or test dependency is required; browser tests use the existing Playwright installation. Browser API fixtures test interaction/layout; transactional behavior is covered separately against SQLite. Real-device Safari and model-dependent native integration tests are not part of these checks.
