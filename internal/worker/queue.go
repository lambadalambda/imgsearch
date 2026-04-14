package worker

import (
	"context"
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"imgsearch/internal/embedder"
	"imgsearch/internal/vectorindex"
)

type Queue struct {
	DB             *sql.DB
	DataDir        string
	LeaseDuration  time.Duration
	RetryBaseDelay time.Duration
	Embedder       embedder.ImageEmbedder
	Annotator      embedder.ImageAnnotator
	Index          vectorindex.VectorIndex
}

type claimedJob struct {
	ID          int64
	ImageID     int64
	ModelID     int64
	Attempts    int
	MaxAttempts int
}

func (q *Queue) ProcessOne(ctx context.Context, owner string) (bool, error) {
	if q == nil || q.DB == nil {
		return false, fmt.Errorf("queue db is nil")
	}
	if q.Embedder == nil {
		return false, fmt.Errorf("queue embedder is nil")
	}
	if q.Index == nil {
		return false, fmt.Errorf("queue index is nil")
	}
	if q.DataDir == "" {
		return false, fmt.Errorf("queue data dir is empty")
	}
	if owner == "" {
		owner = "worker"
	}
	if q.LeaseDuration <= 0 {
		q.LeaseDuration = 30 * time.Second
	}

	job, ok, err := q.claimNext(ctx, owner)
	if err != nil {
		return false, err
	}
	if !ok {
		return false, nil
	}
	log.Printf("worker claimed job id=%d image_id=%d attempt=%d/%d", job.ID, job.ImageID, job.Attempts, job.MaxAttempts)

	storagePath, needsAnnotations, err := q.loadImageTaskData(ctx, job.ImageID)
	if err != nil {
		_ = q.failOrRetry(ctx, job, err)
		return true, err
	}

	absPath := filepath.Join(q.DataDir, filepath.FromSlash(storagePath))
	if _, err := os.Stat(absPath); err != nil {
		_ = q.failOrRetry(ctx, job, fmt.Errorf("stat image file: %w", err))
		return true, err
	}

	vec, err := q.Embedder.EmbedImage(ctx, absPath)
	if err != nil {
		_ = q.failOrRetry(ctx, job, err)
		return true, nil
	}

	if len(vec) == 0 {
		err := fmt.Errorf("empty embedding vector")
		_ = q.failOrRetry(ctx, job, err)
		return true, nil
	}

	var annotation *embedder.ImageAnnotation
	if q.Annotator != nil && needsAnnotations {
		generated, err := q.Annotator.AnnotateImage(ctx, absPath)
		if err != nil {
			log.Printf("worker annotation skipped image_id=%d: %v", job.ImageID, err)
		} else {
			annotation = &generated
		}
	}

	if err := q.completeJob(ctx, job, vec, annotation); err != nil {
		_ = q.failOrRetry(ctx, job, err)
		return true, err
	}

	if err := q.Index.Upsert(ctx, job.ImageID, job.ModelID, vec); err != nil {
		_ = q.failOrRetry(ctx, job, err)
		return true, err
	}
	log.Printf("worker completed job id=%d image_id=%d", job.ID, job.ImageID)

	return true, nil
}

func (q *Queue) claimNext(ctx context.Context, owner string) (claimedJob, bool, error) {
	tx, err := q.DB.BeginTx(ctx, nil)
	if err != nil {
		return claimedJob{}, false, fmt.Errorf("begin claim tx: %w", err)
	}

	var job claimedJob
	err = tx.QueryRowContext(ctx, `
SELECT id, image_id, model_id, attempts, max_attempts
FROM index_jobs
WHERE kind = 'embed_image'
  AND (run_after IS NULL OR run_after <= datetime('now'))
  AND (
    state = 'pending'
    OR (state = 'leased' AND leased_until IS NOT NULL AND leased_until <= datetime('now'))
  )
ORDER BY created_at ASC
LIMIT 1
`).Scan(&job.ID, &job.ImageID, &job.ModelID, &job.Attempts, &job.MaxAttempts)
	if err == sql.ErrNoRows {
		_ = tx.Rollback()
		return claimedJob{}, false, nil
	}
	if err != nil {
		_ = tx.Rollback()
		if isSQLiteLockError(err) {
			return claimedJob{}, false, nil
		}
		return claimedJob{}, false, fmt.Errorf("select claimable job: %w", err)
	}

	leaseSeconds := int(q.LeaseDuration.Seconds())
	if leaseSeconds <= 0 {
		leaseSeconds = 30
	}

	res, err := tx.ExecContext(ctx, `
UPDATE index_jobs
SET state = 'leased',
    attempts = attempts + 1,
    lease_owner = ?,
    leased_until = datetime('now', ?),
    updated_at = datetime('now')
WHERE id = ?
  AND (
    state = 'pending'
    OR (state = 'leased' AND leased_until IS NOT NULL AND leased_until <= datetime('now'))
  )
`, owner, fmt.Sprintf("+%d seconds", leaseSeconds), job.ID)
	if err != nil {
		_ = tx.Rollback()
		if isSQLiteLockError(err) {
			return claimedJob{}, false, nil
		}
		return claimedJob{}, false, fmt.Errorf("update claimed job: %w", err)
	}

	rows, err := res.RowsAffected()
	if err != nil {
		_ = tx.Rollback()
		return claimedJob{}, false, fmt.Errorf("rows affected claim job: %w", err)
	}
	if rows == 0 {
		_ = tx.Rollback()
		return claimedJob{}, false, nil
	}

	if err := tx.Commit(); err != nil {
		return claimedJob{}, false, fmt.Errorf("commit claim tx: %w", err)
	}

	job.Attempts++
	return job, true, nil
}

func (q *Queue) loadImageTaskData(ctx context.Context, imageID int64) (string, bool, error) {
	var storagePath string
	var needsAnnotations int
	if err := q.DB.QueryRowContext(ctx, `
SELECT storage_path,
  CASE
    WHEN trim(COALESCE(description, '')) = ''
      OR COALESCE(tags_json, '') = ''
      OR COALESCE(tags_json, '[]') = '[]'
    THEN 1 ELSE 0
  END
FROM images
WHERE id = ?
`, imageID).Scan(&storagePath, &needsAnnotations); err != nil {
		return "", false, fmt.Errorf("load image task data: %w", err)
	}
	return storagePath, needsAnnotations == 1, nil
}

func (q *Queue) completeJob(ctx context.Context, job claimedJob, vec []float32, annotation *embedder.ImageAnnotation) error {
	tx, err := q.DB.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin complete tx: %w", err)
	}

	blob := floatsToBlob(vec)
	_, err = tx.ExecContext(ctx, `
INSERT INTO image_embeddings(image_id, model_id, dim, vector_blob)
VALUES (?, ?, ?, ?)
ON CONFLICT(image_id, model_id)
DO UPDATE SET
  dim = excluded.dim,
  vector_blob = excluded.vector_blob,
  updated_at = datetime('now')
`, job.ImageID, job.ModelID, len(vec), blob)
	if err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("upsert image embedding: %w", err)
	}

	if annotation != nil {
		tagsJSON, err := json.Marshal(annotation.Tags)
		if err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("marshal image tags: %w", err)
		}
		if _, err := tx.ExecContext(ctx, `
UPDATE images
SET description = ?, tags_json = ?
WHERE id = ?
`, annotation.Description, string(tagsJSON), job.ImageID); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("update image annotations: %w", err)
		}
	}

	if _, err := tx.ExecContext(ctx, `
UPDATE index_jobs
SET state = 'done', leased_until = NULL, lease_owner = NULL, last_error = NULL, updated_at = datetime('now')
WHERE id = ?
`, job.ID); err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("mark job done: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit complete tx: %w", err)
	}

	return nil
}

func (q *Queue) failOrRetry(ctx context.Context, job claimedJob, procErr error) error {
	nextState := "pending"
	if job.Attempts >= job.MaxAttempts {
		nextState = "failed"
	}

	runAfter := sql.NullString{}
	if nextState == "pending" && q.RetryBaseDelay > 0 {
		delay := q.RetryBaseDelay
		if job.Attempts > 1 {
			mul := 1 << (job.Attempts - 1)
			delay = delay * time.Duration(mul)
		}
		maxDelay := 5 * time.Minute
		if delay > maxDelay {
			delay = maxDelay
		}
		runAfter = sql.NullString{String: fmt.Sprintf("+%d seconds", int(delay.Seconds())), Valid: true}
	}

	_, err := q.DB.ExecContext(ctx, `
UPDATE index_jobs
SET state = ?,
    run_after = CASE WHEN ? <> '' THEN datetime('now', ?) ELSE NULL END,
    leased_until = NULL,
    lease_owner = NULL,
    last_error = ?,
    updated_at = datetime('now')
WHERE id = ?
`, nextState, runAfter.String, runAfter.String, procErr.Error(), job.ID)
	if err != nil {
		return fmt.Errorf("update failed job: %w", err)
	}
	log.Printf("worker job id=%d image_id=%d state=%s err=%v", job.ID, job.ImageID, nextState, procErr)
	return nil
}

func floatsToBlob(values []float32) []byte {
	blob := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(blob[i*4:], math.Float32bits(v))
	}
	return blob
}

func isSQLiteLockError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "database is locked") || strings.Contains(msg, "database is busy")
}
