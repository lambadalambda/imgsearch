package upload

import (
	"context"
	"database/sql"
	"fmt"

	"imgsearch/internal/phash"
)

const strictDuplicateDistance = 2

// consolidateStrict runs only after inserting new content, which also acquires
// SQLite's writer reservation. Selection and replacement are thus serialized
// across API processes. Never merge an existing transitive duplicate group.
func consolidateStrict(ctx context.Context, tx *sql.Tx, incoming StoreResult, hash int64, width, height int) (StoreResult, error) {
	if hash == phash.Unhashable || width <= 0 || height <= 0 {
		return incoming, nil
	}
	rows, err := tx.QueryContext(ctx, `
SELECT i.id, i.sha256, i.storage_path, i.phash, i.width * i.height
FROM images i
WHERE i.id != ? AND i.phash IS NOT NULL AND i.phash != ?
  AND i.width > 0 AND i.height > 0
  AND NOT EXISTS (SELECT 1 FROM video_frames vf WHERE vf.image_id = i.id)
ORDER BY i.width * i.height DESC, i.id ASC`, incoming.ImageID, phash.Unhashable)
	if err != nil {
		return StoreResult{}, err
	}
	var existing StoreResult
	var pixels int64
	for rows.Next() {
		var raw int64
		if err := rows.Scan(&existing.ImageID, &existing.SHA256, &existing.StoragePath, &raw, &pixels); err != nil {
			_ = rows.Close()
			return StoreResult{}, err
		}
		if phash.Distance(phash.FromInt64(raw), phash.FromInt64(hash)) <= strictDuplicateDistance {
			break
		}
		existing = StoreResult{}
	}
	err = rows.Err()
	_ = rows.Close()
	if err != nil {
		return StoreResult{}, err
	}
	if existing.ImageID == 0 {
		return incoming, nil
	}

	if pixels >= int64(width)*int64(height) {
		if _, err := tx.ExecContext(ctx, `DELETE FROM images WHERE id = ?`, incoming.ImageID); err != nil {
			return StoreResult{}, err
		}
		existing.MediaType = "image"
		existing.Duplicate = true
		return existing, nil
	}

	// A fresh ID fences old worker results and thumbnail generation without
	// mutating the content behind their IDs. Copy all metadata provenance, but
	// never old embeddings or a thumbnail belonging to the superseded content.
	if _, err := tx.ExecContext(ctx, `
UPDATE images SET
 (title, summary, description, tags_json, annotator_title, annotator_tags_json,
  user_title, user_tags_json, removed_tags_json, reannotate_requested,
  annotation_updated_at, created_at) =
 (SELECT title, summary, description, tags_json, annotator_title, annotator_tags_json,
  user_title, user_tags_json, removed_tags_json, reannotate_requested,
  annotation_updated_at, created_at FROM images WHERE id = ?),
 captured_at = COALESCE(NULLIF(captured_at, ''), (SELECT captured_at FROM images WHERE id = ?))
WHERE id = ?`, existing.ImageID, existing.ImageID, incoming.ImageID); err != nil {
		return StoreResult{}, fmt.Errorf("copy duplicate metadata: %w", err)
	}
	// The new row has no memberships yet: retargeting keeps ordering IDs too.
	if _, err := tx.ExecContext(ctx, `UPDATE collection_items SET image_id = ? WHERE image_id = ?`, incoming.ImageID, existing.ImageID); err != nil {
		return StoreResult{}, fmt.Errorf("transfer duplicate collections: %w", err)
	}
	for _, table := range []string{"image_embeddings", "index_jobs"} {
		if _, err := tx.ExecContext(ctx, `DELETE FROM `+table+` WHERE image_id = ?`, existing.ImageID); err != nil {
			return StoreResult{}, err
		}
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM images WHERE id = ?`, existing.ImageID); err != nil {
		return StoreResult{}, err
	}
	// Retain superseded files: unlinking digest paths after commit can race
	// another import reusing them, and keeping them permits manual recovery.
	incoming.Duplicate = true
	incoming.ReplacedImageID = existing.ImageID
	return incoming, nil
}
