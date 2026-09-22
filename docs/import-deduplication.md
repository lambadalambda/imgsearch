# Unattended strict import deduplication

## Enable

Start the **server/API process** with `-dedup-strict-images` (or `-dedup-strict-images=true`). It is off by default because matching is lossy. Omit the flag or set it to `false` and restart to return to exact-file-only deduplication. There is no per-upload confirmation or importer-side setting.

This applies to all incoming standalone images through `/api/upload`: Atelier uploads, folder imports, and Fediverse imports. It does **not** sweep existing duplicate groups at startup or when enabled. Video uploads and sampled frames continue to use exact-file deduplication only. Unsupported hashes (including AVIF), uncomputed hashes, and unknown dimensions are excluded from perceptual matching.

## Policy

A Strict match means a dHash Hamming distance of at most **2**. It is a lossy visual heuristic, not proof that files or pixels are identical: colors and fine detail can differ. Even distance zero is not exact-file equality. SHA256 remains the first check for identical files; exact re-uploads do not trigger consolidation of existing rows.

For new content, find existing standalone images that directly match the incoming hash, and select the one with the largest pixel count (lowest existing ID breaks ties). Compare that one candidate with the incoming image:

- **Existing is larger or tied:** keep it and discard the incoming temporary file. Return the existing ID and SHA256.
- **Incoming is larger:** replace that one library entry with the incoming image. Transfer collection memberships without changing their saved ordering, plus titles, descriptions, tag additions/removals, annotation provenance, original upload date, and outstanding reannotation requests. Use incoming capture time when present, otherwise preserve the existing capture time.
- **No match:** store normally.

The transitive groups shown in the Duplicates view are not used for unattended consolidation. Other existing members are left alone, so importing one image need not eliminate a pre-existing duplicate group. New larger arrivals can successively replace earlier keepers; comparisons are always against the current keeper, not a permanent historical anchor.

This policy deliberately accepts strict perceptual false positives. It does not claim that a higher-resolution file always has better visual quality.

## Storage and recovery limits

A replacement has a **new image ID**. Collections follow it, but old image-ID API links are not redirected. Re-embedding is queued for the active model; old embeddings for every model and old jobs are removed atomically. Existing annotation text is retained, and missing/requested annotation is queued. A fresh ID also isolates stale worker results and thumbnail generation from the replacement.

**Superseded original and thumbnail files are retained on disk.** This is logical-library deduplication, not a disk-space reclamation job. Retaining old digest paths avoids unsafe unlink races with concurrent imports and allows manual file recovery; it is not automatic metadata undo or a backup system. Incoming smaller/tied copies are not retained. No garbage collector or retroactive cleanup is introduced.

The insert acquires SQLite's writer reservation before selecting a candidate, so imports in separate API processes serialize the selection and replacement. Collection transfer, metadata copy, and job/embedding deletion share the upload transaction. File publication failure restores the old entry; a commit failure can leave an unreferenced new file, which is deliberately retained rather than risking removal of content another upload may reuse. Runtime connections must enable foreign keys (as the application does).

## API

- `duplicate: true` covers both keeping an existing copy and upgrading one.
- `image_id` and `sha256` identify the surviving content, not necessarily the submitted bytes.
- An upgrade also returns `replaced_image_id` and logs the old/new IDs after commit.
- Batch counters describe per-file outcomes: upgrades count as duplicates, not new library additions. When a later file in the same request replaces an earlier result, earlier IDs/digests are updated to that batch's surviving content. `replaced_image_id` still describes the individual historical replacement.
- As with any concurrent deletion, another request can replace an ID after it was returned. Clients should refresh their library rather than treating image IDs as permanent aliases.
