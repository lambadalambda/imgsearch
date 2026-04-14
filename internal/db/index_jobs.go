package db

import (
	"context"
	"database/sql"
	"fmt"
)

func EnsureIndexJobsForModel(ctx context.Context, db *sql.DB, modelID int64) (int64, error) {
	if db == nil {
		return 0, fmt.Errorf("db is nil")
	}
	if modelID <= 0 {
		return 0, fmt.Errorf("invalid model id")
	}

	res, err := db.ExecContext(ctx, `
INSERT OR IGNORE INTO index_jobs(kind, image_id, model_id, state)
SELECT 'embed_image', i.id, ?, 'pending'
FROM images i
`, modelID)
	if err != nil {
		return 0, fmt.Errorf("ensure index jobs for model %d: %w", modelID, err)
	}

	rows, err := res.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("rows affected for model %d: %w", modelID, err)
	}
	return rows, nil
}

func RequeueDoneJobsMissingAnnotations(ctx context.Context, db *sql.DB, modelID int64) (int64, error) {
	if db == nil {
		return 0, fmt.Errorf("db is nil")
	}
	if modelID <= 0 {
		return 0, fmt.Errorf("invalid model id")
	}

	res, err := db.ExecContext(ctx, `
UPDATE index_jobs
SET state = 'pending',
    attempts = 0,
    run_after = NULL,
    leased_until = NULL,
    lease_owner = NULL,
    last_error = NULL,
    updated_at = datetime('now')
WHERE kind = 'embed_image'
  AND model_id = ?
  AND state = 'done'
  AND EXISTS (
    SELECT 1
    FROM images i
    WHERE i.id = index_jobs.image_id
      AND (
        trim(COALESCE(i.description, '')) = ''
        OR COALESCE(i.tags_json, '') = ''
        OR COALESCE(i.tags_json, '[]') = '[]'
      )
  )
`, modelID)
	if err != nil {
		return 0, fmt.Errorf("requeue jobs missing annotations for model %d: %w", modelID, err)
	}

	rows, err := res.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("rows affected while requeueing model %d: %w", modelID, err)
	}
	return rows, nil
}

func RequeueDoneJob(ctx context.Context, db *sql.DB, modelID int64, imageID int64) (bool, error) {
	if db == nil {
		return false, fmt.Errorf("db is nil")
	}
	if modelID <= 0 {
		return false, fmt.Errorf("invalid model id")
	}
	if imageID <= 0 {
		return false, fmt.Errorf("invalid image id")
	}

	res, err := db.ExecContext(ctx, `
UPDATE index_jobs
SET state = 'pending',
    attempts = 0,
    run_after = NULL,
    leased_until = NULL,
    lease_owner = NULL,
    last_error = NULL,
    updated_at = datetime('now')
WHERE kind = 'embed_image'
  AND model_id = ?
  AND image_id = ?
  AND state = 'done'
`, modelID, imageID)
	if err != nil {
		return false, fmt.Errorf("requeue done job for image %d model %d: %w", imageID, modelID, err)
	}

	rows, err := res.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("rows affected while requeueing image %d model %d: %w", imageID, modelID, err)
	}
	return rows > 0, nil
}
