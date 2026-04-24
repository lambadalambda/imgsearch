package mediaops

import (
	"context"
	"database/sql"
	"fmt"

	"imgsearch/internal/jobkind"
)

type ReannotationTarget struct {
	Kind    string
	ImageID int64
	VideoID int64
	ModelID int64
}

func RequestReannotationJob(ctx context.Context, tx *sql.Tx, target ReannotationTarget) error {
	if tx == nil {
		return fmt.Errorf("transaction is nil")
	}
	if target.ModelID <= 0 {
		return fmt.Errorf("invalid model id")
	}

	switch target.Kind {
	case jobkind.AnnotateImage:
		if target.ImageID <= 0 {
			return fmt.Errorf("invalid image id")
		}
		if _, err := tx.ExecContext(ctx, `
INSERT INTO index_jobs(kind, image_id, model_id, state)
VALUES(?, ?, ?, 'pending')
ON CONFLICT DO NOTHING
`, jobkind.AnnotateImage, target.ImageID, target.ModelID); err != nil {
			return fmt.Errorf("insert image annotation job: %w", err)
		}
		if _, err := tx.ExecContext(ctx, `
UPDATE index_jobs
SET state = 'pending',
    attempts = 0,
    run_after = NULL,
    leased_until = NULL,
    lease_owner = NULL,
    last_error = NULL,
    updated_at = datetime('now')
WHERE kind = ?
  AND image_id = ?
  AND model_id = ?
  AND state <> 'leased'
`, jobkind.AnnotateImage, target.ImageID, target.ModelID); err != nil {
			return fmt.Errorf("reset image annotation job: %w", err)
		}
	case jobkind.AnnotateVideo:
		if target.VideoID <= 0 {
			return fmt.Errorf("invalid video id")
		}
		if _, err := tx.ExecContext(ctx, `
INSERT INTO index_jobs(kind, image_id, video_id, model_id, state)
VALUES(?, NULL, ?, ?, 'pending')
ON CONFLICT DO NOTHING
`, jobkind.AnnotateVideo, target.VideoID, target.ModelID); err != nil {
			return fmt.Errorf("insert video annotation job: %w", err)
		}
		if _, err := tx.ExecContext(ctx, `
UPDATE index_jobs
SET state = 'pending',
    attempts = 0,
    run_after = NULL,
    leased_until = NULL,
    lease_owner = NULL,
    last_error = NULL,
    updated_at = datetime('now')
WHERE kind = ?
  AND video_id = ?
  AND model_id = ?
  AND state <> 'leased'
`, jobkind.AnnotateVideo, target.VideoID, target.ModelID); err != nil {
			return fmt.Errorf("reset video annotation job: %w", err)
		}
	default:
		return fmt.Errorf("unsupported annotation job kind")
	}

	return nil
}
