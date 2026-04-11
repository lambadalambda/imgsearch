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
