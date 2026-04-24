package mediaops

import (
	"context"
	"database/sql"
	"fmt"
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
	case "annotate_image":
		if target.ImageID <= 0 {
			return fmt.Errorf("invalid image id")
		}
		if _, err := tx.ExecContext(ctx, `
INSERT INTO index_jobs(kind, image_id, model_id, state)
VALUES('annotate_image', ?, ?, 'pending')
ON CONFLICT DO NOTHING
`, target.ImageID, target.ModelID); err != nil {
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
WHERE kind = 'annotate_image'
  AND image_id = ?
  AND model_id = ?
  AND state <> 'leased'
`, target.ImageID, target.ModelID); err != nil {
			return fmt.Errorf("reset image annotation job: %w", err)
		}
	case "annotate_video":
		if target.VideoID <= 0 {
			return fmt.Errorf("invalid video id")
		}
		if _, err := tx.ExecContext(ctx, `
INSERT INTO index_jobs(kind, image_id, video_id, model_id, state)
VALUES('annotate_video', NULL, ?, ?, 'pending')
ON CONFLICT DO NOTHING
`, target.VideoID, target.ModelID); err != nil {
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
WHERE kind = 'annotate_video'
  AND video_id = ?
  AND model_id = ?
  AND state <> 'leased'
`, target.VideoID, target.ModelID); err != nil {
			return fmt.Errorf("reset video annotation job: %w", err)
		}
	default:
		return fmt.Errorf("unsupported annotation job kind")
	}

	return nil
}
