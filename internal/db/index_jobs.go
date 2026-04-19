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

func EnsureAnnotationJobsForModel(ctx context.Context, db *sql.DB, modelID int64) (int64, error) {
	if db == nil {
		return 0, fmt.Errorf("db is nil")
	}
	if modelID <= 0 {
		return 0, fmt.Errorf("invalid model id")
	}

	res, err := db.ExecContext(ctx, `
INSERT OR IGNORE INTO index_jobs(kind, image_id, model_id, state)
SELECT 'annotate_image', i.id, ?, 'pending'
FROM images i
WHERE NOT EXISTS (
  SELECT 1
  FROM video_frames vf
  WHERE vf.image_id = i.id
)
  AND (
    trim(COALESCE(i.description, '')) = ''
    OR COALESCE(i.tags_json, '') = ''
    OR COALESCE(i.tags_json, '[]') = '[]'
  )
`, modelID)
	if err != nil {
		return 0, fmt.Errorf("ensure annotation jobs for model %d: %w", modelID, err)
	}

	rows, err := res.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("rows affected for annotation jobs model %d: %w", modelID, err)
	}
	return rows, nil
}

func EnsureVideoTranscriptJobsForModel(ctx context.Context, db *sql.DB, modelID int64) (int64, error) {
	if db == nil {
		return 0, fmt.Errorf("db is nil")
	}
	if modelID <= 0 {
		return 0, fmt.Errorf("invalid model id")
	}

	res, err := db.ExecContext(ctx, `
INSERT OR IGNORE INTO index_jobs(kind, image_id, video_id, model_id, state)
SELECT 'transcribe_video', NULL, v.id, ?, 'pending'
FROM videos v
LEFT JOIN video_transcript_embeddings vte
  ON vte.video_id = v.id
 AND vte.model_id = ?
WHERE trim(COALESCE(v.transcript_text, '')) = ''
   OR vte.video_id IS NULL
`, modelID, modelID)
	if err != nil {
		return 0, fmt.Errorf("ensure video transcript jobs for model %d: %w", modelID, err)
	}

	rows, err := res.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("rows affected for video transcript jobs model %d: %w", modelID, err)
	}
	return rows, nil
}

func EnsureVideoAnnotationJobsForModel(ctx context.Context, db *sql.DB, modelID int64) (int64, error) {
	if db == nil {
		return 0, fmt.Errorf("db is nil")
	}
	if modelID <= 0 {
		return 0, fmt.Errorf("invalid model id")
	}

	res, err := db.ExecContext(ctx, `
INSERT OR IGNORE INTO index_jobs(kind, image_id, video_id, model_id, state)
SELECT 'annotate_video', NULL, v.id, ?, 'pending'
FROM videos v
WHERE trim(COALESCE(v.description, '')) = ''
   OR COALESCE(v.tags_json, '') = ''
   OR COALESCE(v.tags_json, '[]') = '[]'
`, modelID)
	if err != nil {
		return 0, fmt.Errorf("ensure video annotation jobs for model %d: %w", modelID, err)
	}

	rows, err := res.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("rows affected for video annotation jobs model %d: %w", modelID, err)
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
WHERE kind = 'annotate_image'
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
  AND NOT EXISTS (
    SELECT 1
    FROM video_frames vf
    WHERE vf.image_id = index_jobs.image_id
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

func PurgeOtherModelIndexJobs(ctx context.Context, db *sql.DB, activeModelID int64) (int64, error) {
	if db == nil {
		return 0, fmt.Errorf("db is nil")
	}
	if activeModelID <= 0 {
		return 0, fmt.Errorf("invalid active model id")
	}

	res, err := db.ExecContext(ctx, `DELETE FROM index_jobs WHERE model_id <> ?`, activeModelID)
	if err != nil {
		return 0, fmt.Errorf("purge index jobs for other models: %w", err)
	}
	rows, err := res.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("rows affected while purging index jobs: %w", err)
	}
	return rows, nil
}

func PurgeOtherModelVideoTranscriptEmbeddings(ctx context.Context, db *sql.DB, activeModelID int64) (int64, error) {
	if db == nil {
		return 0, fmt.Errorf("db is nil")
	}
	if activeModelID <= 0 {
		return 0, fmt.Errorf("invalid active model id")
	}

	res, err := db.ExecContext(ctx, `DELETE FROM video_transcript_embeddings WHERE model_id <> ?`, activeModelID)
	if err != nil {
		return 0, fmt.Errorf("purge video transcript embeddings for other models: %w", err)
	}
	rows, err := res.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("rows affected while purging video transcript embeddings: %w", err)
	}
	return rows, nil
}
