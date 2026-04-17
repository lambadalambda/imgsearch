package worker

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"imgsearch/internal/embedder"
	"imgsearch/internal/transcribe"
	"imgsearch/internal/vectorindex"
)

type Queue struct {
	DB             *sql.DB
	DataDir        string
	LeaseDuration  time.Duration
	RetryBaseDelay time.Duration
	Embedder       embedder.ImageEmbedder
	TextEmbedder   embedder.TextEmbedder
	Annotator      embedder.ImageAnnotator
	Transcriber    transcribe.VideoTranscriber
	Index          vectorindex.VectorIndex
}

type claimedJob struct {
	ID          int64
	Kind        string
	ImageID     int64
	VideoID     int64
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
	log.Printf("worker claimed job id=%d kind=%s image_id=%d attempt=%d/%d", job.ID, job.Kind, job.ImageID, job.Attempts, job.MaxAttempts)
	jobStartedAt := time.Now()
	var loadTaskDataDuration time.Duration
	var statDuration time.Duration
	var embedDuration time.Duration
	var transcribeDuration time.Duration
	var annotateDuration time.Duration
	var completeDuration time.Duration
	var indexDuration time.Duration
	var stageStartedAt time.Time

	storagePath := ""
	needsAnnotations := false
	absPath := ""
	if job.Kind != "transcribe_video" {
		stageStartedAt := time.Now()
		storagePath, needsAnnotations, err = q.loadImageTaskData(ctx, job.ImageID)
		loadTaskDataDuration = time.Since(stageStartedAt)
		if err != nil {
			_ = q.failOrRetry(ctx, job, err)
			return true, err
		}

		absPath = filepath.Join(q.DataDir, filepath.FromSlash(storagePath))
		stageStartedAt = time.Now()
		if _, err := os.Stat(absPath); err != nil {
			statDuration = time.Since(stageStartedAt)
			_ = q.failOrRetry(ctx, job, fmt.Errorf("stat image file: %w", err))
			return true, err
		}
		statDuration = time.Since(stageStartedAt)
	}

	var annotation *embedder.ImageAnnotation
	annotationAttempted := false
	annotationStored := false
	annotationErrText := ""

	switch job.Kind {
	case "embed_image":
		stageStartedAt = time.Now()
		vec, err := q.Embedder.EmbedImage(ctx, absPath)
		embedDuration = time.Since(stageStartedAt)
		if err != nil {
			_ = q.failOrRetry(ctx, job, err)
			return true, nil
		}

		if len(vec) == 0 {
			err := fmt.Errorf("empty embedding vector")
			_ = q.failOrRetry(ctx, job, err)
			return true, nil
		}

		stageStartedAt = time.Now()
		if err := q.Index.Upsert(ctx, job.ImageID, job.ModelID, vec); err != nil {
			indexDuration = time.Since(stageStartedAt)
			_ = q.failOrRetry(ctx, job, err)
			return true, err
		}
		indexDuration = time.Since(stageStartedAt)
	case "annotate_image":
		if !needsAnnotations {
			break
		}
		if q.Annotator == nil {
			return false, nil
		}
		annotationAttempted = true
		stageStartedAt = time.Now()
		generated, err := q.Annotator.AnnotateImage(ctx, absPath)
		annotateDuration = time.Since(stageStartedAt)
		if err != nil {
			annotationErrText = err.Error()
			_ = q.failOrRetry(ctx, job, err)
			return true, nil
		}
		annotation = &generated
		annotationStored = true
	case "transcribe_video":
		if q.Transcriber == nil {
			err := fmt.Errorf("video transcriber is unavailable")
			_ = q.failOrRetry(ctx, job, err)
			return true, err
		}
		if q.TextEmbedder == nil {
			err := fmt.Errorf("text embedder is unavailable")
			_ = q.failOrRetry(ctx, job, err)
			return true, err
		}
		videoID, videoOriginalName, videoStoragePath, err := q.loadVideoTaskData(ctx, job.VideoID)
		if err != nil {
			_ = q.failOrRetry(ctx, job, err)
			return true, err
		}
		videoAbsPath := filepath.Join(q.DataDir, filepath.FromSlash(videoStoragePath))
		log.Printf("worker transcribing video id=%d file=%q path=%s", videoID, videoOriginalName, videoStoragePath)
		stageStartedAt = time.Now()
		if _, err := os.Stat(videoAbsPath); err != nil {
			statDuration = time.Since(stageStartedAt)
			_ = q.failOrRetry(ctx, job, fmt.Errorf("stat video file: %w", err))
			return true, err
		}
		statDuration = time.Since(stageStartedAt)

		stageStartedAt = time.Now()
		transcript, err := q.Transcriber.TranscribeVideo(ctx, videoAbsPath)
		transcribeDuration = time.Since(stageStartedAt)
		if err != nil {
			_ = q.failOrRetry(ctx, job, err)
			return true, nil
		}
		if strings.TrimSpace(transcript.Text) == "" {
			err := fmt.Errorf("empty transcript")
			_ = q.failOrRetry(ctx, job, err)
			return true, nil
		}

		stageStartedAt = time.Now()
		vec, err := q.TextEmbedder.EmbedText(ctx, transcript.Text)
		embedDuration = time.Since(stageStartedAt)
		if err != nil {
			_ = q.failOrRetry(ctx, job, err)
			return true, nil
		}
		if len(vec) == 0 {
			err := fmt.Errorf("empty transcript embedding vector")
			_ = q.failOrRetry(ctx, job, err)
			return true, nil
		}

		stageStartedAt = time.Now()
		if err := q.completeVideoTranscriptJob(ctx, job, videoID, transcript, vec); err != nil {
			completeDuration = time.Since(stageStartedAt)
			_ = q.failOrRetry(ctx, job, err)
			return true, err
		}
		completeDuration = time.Since(stageStartedAt)
	default:
		err := fmt.Errorf("unsupported job kind %q", job.Kind)
		_ = q.failOrRetry(ctx, job, err)
		return true, err
	}

	if job.Kind != "transcribe_video" {
		stageStartedAt = time.Now()
		if err := q.completeJob(ctx, job, annotation); err != nil {
			completeDuration = time.Since(stageStartedAt)
			_ = q.failOrRetry(ctx, job, err)
			return true, err
		}
		completeDuration = time.Since(stageStartedAt)
	}

	annotationErrSuffix := ""
	if annotationErrText != "" {
		annotationErrSuffix = fmt.Sprintf(" annotation_error=%q", annotationErrText)
	}
	log.Printf(
		"worker completed job id=%d kind=%s image_id=%d total=%s load=%s stat=%s embed=%s transcribe=%s annotate=%s db=%s index=%s annotations_needed=%t annotation_attempted=%t annotation_stored=%t%s",
		job.ID,
		job.Kind,
		job.ImageID,
		time.Since(jobStartedAt).Round(time.Millisecond),
		loadTaskDataDuration.Round(time.Millisecond),
		statDuration.Round(time.Millisecond),
		embedDuration.Round(time.Millisecond),
		transcribeDuration.Round(time.Millisecond),
		annotateDuration.Round(time.Millisecond),
		completeDuration.Round(time.Millisecond),
		indexDuration.Round(time.Millisecond),
		needsAnnotations,
		annotationAttempted,
		annotationStored,
		annotationErrSuffix,
	)

	return true, nil
}

func (q *Queue) claimNext(ctx context.Context, owner string) (claimedJob, bool, error) {
	tx, err := q.DB.BeginTx(ctx, nil)
	if err != nil {
		return claimedJob{}, false, fmt.Errorf("begin claim tx: %w", err)
	}

	kindFilter, orderBy := q.claimableKindSQL()

	var job claimedJob
	err = tx.QueryRowContext(ctx, fmt.Sprintf(`
SELECT id, kind, COALESCE(image_id, 0), COALESCE(video_id, 0), model_id, attempts, max_attempts
FROM index_jobs
WHERE %s
  AND (run_after IS NULL OR run_after <= datetime('now'))
  AND (
    state = 'pending'
    OR (state = 'leased' AND leased_until IS NOT NULL AND leased_until <= datetime('now'))
  )
ORDER BY %s
LIMIT 1
`, kindFilter, orderBy)).Scan(&job.ID, &job.Kind, &job.ImageID, &job.VideoID, &job.ModelID, &job.Attempts, &job.MaxAttempts)
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

func (q *Queue) claimBatch(ctx context.Context, owner string, limit int) ([]claimedJob, error) {
	if limit <= 0 {
		limit = 1
	}

	tx, err := q.DB.BeginTx(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("begin claim batch tx: %w", err)
	}

	kindFilter, orderBy := q.claimableKindSQL()

	rows, err := tx.QueryContext(ctx, fmt.Sprintf(`
SELECT id, kind FROM index_jobs
WHERE %s
  AND (run_after IS NULL OR run_after <= datetime('now'))
  AND (
    state = 'pending'
    OR (state = 'leased' AND leased_until IS NOT NULL AND leased_until <= datetime('now'))
  )
ORDER BY %s
LIMIT ?
`, kindFilter, orderBy), limit)
	if err != nil {
		_ = tx.Rollback()
		if isSQLiteLockError(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("select claimable jobs: %w", err)
	}

	type idKind struct {
		id   int64
		kind string
	}
	var candidates []idKind
	for rows.Next() {
		var ik idKind
		if err := rows.Scan(&ik.id, &ik.kind); err != nil {
			_ = rows.Close()
			_ = tx.Rollback()
			return nil, fmt.Errorf("scan claimable job: %w", err)
		}
		candidates = append(candidates, ik)
	}
	_ = rows.Close()

	if len(candidates) == 0 {
		_ = tx.Rollback()
		return nil, nil
	}

	chosenKind := candidates[0].kind
	var filtered []idKind
	for _, c := range candidates {
		if c.kind == chosenKind {
			filtered = append(filtered, c)
		}
	}

	leaseSeconds := int(q.LeaseDuration.Seconds())
	if leaseSeconds <= 0 {
		leaseSeconds = 30
	}

	placeholders := make([]string, len(filtered))
	args := make([]any, 0, len(filtered)*3+1)
	for i, f := range filtered {
		placeholders[i] = "?"
		args = append(args, owner, fmt.Sprintf("+%d seconds", leaseSeconds), f.id)
	}

	query := fmt.Sprintf(`
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
`)
	var updateQuery strings.Builder
	updateQuery.WriteString("UPDATE index_jobs SET state = 'leased', attempts = attempts + 1, lease_owner = ?, leased_until = datetime('now', ?), updated_at = datetime('now') WHERE id = ? AND (state = 'pending' OR (state = 'leased' AND leased_until IS NOT NULL AND leased_until <= datetime('now')))")
	for i := 1; i < len(filtered); i++ {
		updateQuery.WriteString(";\n")
		updateQuery.WriteString(query)
		args = append(args, owner, fmt.Sprintf("+%d seconds", leaseSeconds), filtered[i].id)
	}

	_, err = tx.ExecContext(ctx, updateQuery.String(), args...)
	if err != nil {
		_ = tx.Rollback()
		if isSQLiteLockError(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("update claimed batch: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("commit claim batch tx: %w", err)
	}

	idArgs := make([]any, len(filtered))
	for i, f := range filtered {
		idArgs[i] = f.id
	}
	ph := make([]string, len(filtered))
	for i := range ph {
		ph[i] = "?"
	}

	var jobs []claimedJob
	jobRows, err := q.DB.QueryContext(ctx, fmt.Sprintf(`
SELECT id, kind, COALESCE(image_id, 0), COALESCE(video_id, 0), model_id, attempts, max_attempts
FROM index_jobs
WHERE id IN (%s)
ORDER BY created_at ASC
`, strings.Join(ph, ",")), idArgs...)
	if err != nil {
		return nil, fmt.Errorf("select claimed jobs: %w", err)
	}
	for jobRows.Next() {
		var j claimedJob
		if err := jobRows.Scan(&j.ID, &j.Kind, &j.ImageID, &j.VideoID, &j.ModelID, &j.Attempts, &j.MaxAttempts); err != nil {
			_ = jobRows.Close()
			return nil, fmt.Errorf("scan claimed job: %w", err)
		}
		jobs = append(jobs, j)
	}
	_ = jobRows.Close()

	return jobs, nil
}

func (q *Queue) ProcessBatch(ctx context.Context, owner string, batchSize int) (int, error) {
	if q == nil || q.DB == nil {
		return 0, fmt.Errorf("queue db is nil")
	}
	if q.Embedder == nil {
		return 0, fmt.Errorf("queue embedder is nil")
	}
	if q.Index == nil {
		return 0, fmt.Errorf("queue index is nil")
	}
	if q.DataDir == "" {
		return 0, fmt.Errorf("queue data dir is empty")
	}
	if batchSize <= 0 {
		batchSize = 1
	}

	jobs, err := q.claimBatch(ctx, owner, batchSize)
	if err != nil {
		return 0, err
	}
	if len(jobs) == 0 {
		return 0, nil
	}
	if owner == "" {
		owner = "worker"
	}
	if q.LeaseDuration <= 0 {
		q.LeaseDuration = 30 * time.Second
	}

	if len(jobs) > 1 {
		if batcher, ok := q.Embedder.(embedder.BatchImageEmbedder); ok && allClaimedJobsAreKind(jobs, "embed_image") {
			return q.processEmbedJobsBatch(ctx, batcher, jobs), nil
		}
	}

	processed := 0
	for _, job := range jobs {
		if err := ctx.Err(); err != nil {
			return processed, err
		}
		if q.processJob(ctx, owner, job) {
			processed++
		}
	}
	return processed, nil
}

func allClaimedJobsAreKind(jobs []claimedJob, kind string) bool {
	if len(jobs) == 0 {
		return false
	}
	for _, job := range jobs {
		if job.Kind != kind {
			return false
		}
	}
	return true
}

func (q *Queue) processEmbedJobsBatch(ctx context.Context, batcher embedder.BatchImageEmbedder, jobs []claimedJob) int {
	type preparedJob struct {
		job              claimedJob
		absPath          string
		needsAnnotations bool
	}

	prepared := make([]preparedJob, 0, len(jobs))
	for _, job := range jobs {
		log.Printf("worker claimed job id=%d kind=%s image_id=%d attempt=%d/%d", job.ID, job.Kind, job.ImageID, job.Attempts, job.MaxAttempts)

		storagePath, needsAnnotations, err := q.loadImageTaskData(ctx, job.ImageID)
		if err != nil {
			_ = q.failOrRetry(ctx, job, err)
			continue
		}

		absPath := filepath.Join(q.DataDir, filepath.FromSlash(storagePath))
		if _, err := os.Stat(absPath); err != nil {
			_ = q.failOrRetry(ctx, job, fmt.Errorf("stat image file: %w", err))
			continue
		}

		prepared = append(prepared, preparedJob{
			job:              job,
			absPath:          absPath,
			needsAnnotations: needsAnnotations,
		})
	}

	if len(prepared) == 0 {
		return 0
	}

	paths := make([]string, len(prepared))
	for i, item := range prepared {
		paths[i] = item.absPath
	}

	batchStartedAt := time.Now()
	vecs, err := batcher.EmbedImages(ctx, paths)
	batchDuration := time.Since(batchStartedAt)
	if err != nil {
		for _, item := range prepared {
			_ = q.failOrRetry(ctx, item.job, err)
		}
		return 0
	}
	if len(vecs) != len(prepared) {
		err := fmt.Errorf("batch embedding count mismatch: got %d want %d", len(vecs), len(prepared))
		for _, item := range prepared {
			_ = q.failOrRetry(ctx, item.job, err)
		}
		return 0
	}

	avgEmbedDuration := batchDuration / time.Duration(len(prepared))
	processed := 0
	for i, item := range prepared {
		if len(vecs[i]) == 0 {
			_ = q.failOrRetry(ctx, item.job, fmt.Errorf("empty embedding vector"))
			continue
		}

		indexStartedAt := time.Now()
		if err := q.Index.Upsert(ctx, item.job.ImageID, item.job.ModelID, vecs[i]); err != nil {
			_ = q.failOrRetry(ctx, item.job, err)
			continue
		}
		indexDuration := time.Since(indexStartedAt)

		completeStartedAt := time.Now()
		if err := q.completeJob(ctx, item.job, nil); err != nil {
			_ = q.failOrRetry(ctx, item.job, err)
			continue
		}
		completeDuration := time.Since(completeStartedAt)

		log.Printf(
			"worker completed job id=%d kind=%s image_id=%d total=%s load=%s stat=%s embed=%s annotate=%s db=%s index=%s annotations_needed=%t annotation_attempted=%t annotation_stored=%t batch_size=%d",
			item.job.ID,
			item.job.Kind,
			item.job.ImageID,
			batchDuration.Round(time.Millisecond),
			0*time.Millisecond,
			0*time.Millisecond,
			avgEmbedDuration.Round(time.Millisecond),
			0*time.Millisecond,
			completeDuration.Round(time.Millisecond),
			indexDuration.Round(time.Millisecond),
			item.needsAnnotations,
			false,
			false,
			len(prepared),
		)
		processed++
	}

	return processed
}

func (q *Queue) processJob(ctx context.Context, owner string, job claimedJob) bool {
	log.Printf("worker claimed job id=%d kind=%s image_id=%d attempt=%d/%d", job.ID, job.Kind, job.ImageID, job.Attempts, job.MaxAttempts)
	jobStartedAt := time.Now()
	var loadTaskDataDuration time.Duration
	var statDuration time.Duration
	var embedDuration time.Duration
	var transcribeDuration time.Duration
	var annotateDuration time.Duration
	var completeDuration time.Duration
	var indexDuration time.Duration
	var stageStartedAt time.Time
	var err error

	storagePath := ""
	needsAnnotations := false
	absPath := ""
	if job.Kind != "transcribe_video" {
		stageStartedAt := time.Now()
		storagePath, needsAnnotations, err = q.loadImageTaskData(ctx, job.ImageID)
		loadTaskDataDuration = time.Since(stageStartedAt)
		if err != nil {
			_ = q.failOrRetry(ctx, job, err)
			return false
		}

		absPath = filepath.Join(q.DataDir, filepath.FromSlash(storagePath))
		stageStartedAt = time.Now()
		if _, err := os.Stat(absPath); err != nil {
			statDuration = time.Since(stageStartedAt)
			_ = q.failOrRetry(ctx, job, fmt.Errorf("stat image file: %w", err))
			return false
		}
		statDuration = time.Since(stageStartedAt)
	}

	var annotation *embedder.ImageAnnotation
	annotationAttempted := false
	annotationStored := false
	annotationErrText := ""

	switch job.Kind {
	case "embed_image":
		stageStartedAt = time.Now()
		vec, err := q.Embedder.EmbedImage(ctx, absPath)
		embedDuration = time.Since(stageStartedAt)
		if err != nil {
			_ = q.failOrRetry(ctx, job, err)
			return false
		}

		if len(vec) == 0 {
			_ = q.failOrRetry(ctx, job, fmt.Errorf("empty embedding vector"))
			return false
		}

		stageStartedAt = time.Now()
		if err := q.Index.Upsert(ctx, job.ImageID, job.ModelID, vec); err != nil {
			indexDuration = time.Since(stageStartedAt)
			_ = q.failOrRetry(ctx, job, err)
			return false
		}
		indexDuration = time.Since(stageStartedAt)
	case "annotate_image":
		if !needsAnnotations {
			break
		}
		if q.Annotator == nil {
			return false
		}
		annotationAttempted = true
		stageStartedAt = time.Now()
		generated, err := q.Annotator.AnnotateImage(ctx, absPath)
		annotateDuration = time.Since(stageStartedAt)
		if err != nil {
			annotationErrText = err.Error()
			_ = q.failOrRetry(ctx, job, err)
			return false
		}
		annotation = &generated
		annotationStored = true
	case "transcribe_video":
		if q.Transcriber == nil {
			_ = q.failOrRetry(ctx, job, fmt.Errorf("video transcriber is unavailable"))
			return false
		}
		if q.TextEmbedder == nil {
			_ = q.failOrRetry(ctx, job, fmt.Errorf("text embedder is unavailable"))
			return false
		}
		videoID, videoOriginalName, videoStoragePath, err := q.loadVideoTaskData(ctx, job.VideoID)
		if err != nil {
			_ = q.failOrRetry(ctx, job, err)
			return false
		}
		videoAbsPath := filepath.Join(q.DataDir, filepath.FromSlash(videoStoragePath))
		log.Printf("worker transcribing video id=%d file=%q path=%s", videoID, videoOriginalName, videoStoragePath)
		stageStartedAt = time.Now()
		if _, err := os.Stat(videoAbsPath); err != nil {
			statDuration = time.Since(stageStartedAt)
			_ = q.failOrRetry(ctx, job, fmt.Errorf("stat video file: %w", err))
			return false
		}
		statDuration = time.Since(stageStartedAt)

		stageStartedAt = time.Now()
		transcript, err := q.Transcriber.TranscribeVideo(ctx, videoAbsPath)
		transcribeDuration = time.Since(stageStartedAt)
		if err != nil {
			_ = q.failOrRetry(ctx, job, err)
			return false
		}
		if strings.TrimSpace(transcript.Text) == "" {
			_ = q.failOrRetry(ctx, job, fmt.Errorf("empty transcript"))
			return false
		}

		stageStartedAt = time.Now()
		vec, err := q.TextEmbedder.EmbedText(ctx, transcript.Text)
		embedDuration = time.Since(stageStartedAt)
		if err != nil {
			_ = q.failOrRetry(ctx, job, err)
			return false
		}
		if len(vec) == 0 {
			_ = q.failOrRetry(ctx, job, fmt.Errorf("empty transcript embedding vector"))
			return false
		}

		stageStartedAt = time.Now()
		if err := q.completeVideoTranscriptJob(ctx, job, videoID, transcript, vec); err != nil {
			completeDuration = time.Since(stageStartedAt)
			_ = q.failOrRetry(ctx, job, err)
			return false
		}
		completeDuration = time.Since(stageStartedAt)
	default:
		_ = q.failOrRetry(ctx, job, fmt.Errorf("unsupported job kind %q", job.Kind))
		return false
	}

	if job.Kind != "transcribe_video" {
		stageStartedAt = time.Now()
		if err := q.completeJob(ctx, job, annotation); err != nil {
			completeDuration = time.Since(stageStartedAt)
			_ = q.failOrRetry(ctx, job, err)
			return false
		}
		completeDuration = time.Since(stageStartedAt)
	}

	annotationErrSuffix := ""
	if annotationErrText != "" {
		annotationErrSuffix = fmt.Sprintf(" annotation_error=%q", annotationErrText)
	}
	log.Printf(
		"worker completed job id=%d kind=%s image_id=%d total=%s load=%s stat=%s embed=%s transcribe=%s annotate=%s db=%s index=%s annotations_needed=%t annotation_attempted=%t annotation_stored=%t%s",
		job.ID,
		job.Kind,
		job.ImageID,
		time.Since(jobStartedAt).Round(time.Millisecond),
		loadTaskDataDuration.Round(time.Millisecond),
		statDuration.Round(time.Millisecond),
		embedDuration.Round(time.Millisecond),
		transcribeDuration.Round(time.Millisecond),
		annotateDuration.Round(time.Millisecond),
		completeDuration.Round(time.Millisecond),
		indexDuration.Round(time.Millisecond),
		needsAnnotations,
		annotationAttempted,
		annotationStored,
		annotationErrSuffix,
	)

	return true
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

func (q *Queue) loadVideoTaskData(ctx context.Context, videoID int64) (int64, string, string, error) {
	var originalName string
	var storagePath string
	if err := q.DB.QueryRowContext(ctx, `
SELECT id, original_name, storage_path
FROM videos
WHERE id = ?
`, videoID).Scan(&videoID, &originalName, &storagePath); err != nil {
		return 0, "", "", fmt.Errorf("load video task data: %w", err)
	}
	return videoID, originalName, storagePath, nil
}

func (q *Queue) completeJob(ctx context.Context, job claimedJob, annotation *embedder.ImageAnnotation) error {
	tx, err := q.DB.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin complete tx: %w", err)
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

func (q *Queue) completeVideoTranscriptJob(ctx context.Context, job claimedJob, videoID int64, transcript transcribe.Transcript, vec []float32) error {
	tx, err := q.DB.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin complete video transcript tx: %w", err)
	}

	if _, err := tx.ExecContext(ctx, `
UPDATE videos
SET transcript_text = ?, transcript_updated_at = datetime('now')
WHERE id = ?
`, transcript.Text, videoID); err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("update video transcript: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `
INSERT INTO video_transcript_embeddings(video_id, model_id, dim, vector_blob)
VALUES (?, ?, ?, ?)
ON CONFLICT(video_id, model_id)
DO UPDATE SET dim = excluded.dim, vector_blob = excluded.vector_blob, updated_at = datetime('now')
`, videoID, job.ModelID, len(vec), vectorindex.FloatsToBlob(vec)); err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("upsert video transcript embedding: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `
UPDATE index_jobs
SET state = 'done', leased_until = NULL, lease_owner = NULL, last_error = NULL, updated_at = datetime('now')
WHERE id = ?
`, job.ID); err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("mark transcript job done: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit complete video transcript tx: %w", err)
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
func isSQLiteLockError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "database is locked") || strings.Contains(msg, "database is busy")
}

func (q *Queue) claimableKindSQL() (string, string) {
	kinds := []string{"'embed_image'"}
	orderParts := []string{"WHEN 'embed_image' THEN 0"}
	priority := 1
	if q.Transcriber != nil {
		kinds = append(kinds, "'transcribe_video'")
		orderParts = append(orderParts, fmt.Sprintf("WHEN 'transcribe_video' THEN %d", priority))
		priority++
	}
	if q.Annotator != nil {
		kinds = append(kinds, "'annotate_image'")
		orderParts = append(orderParts, fmt.Sprintf("WHEN 'annotate_image' THEN %d", priority))
		priority++
	}
	return fmt.Sprintf("kind IN (%s)", strings.Join(kinds, ", ")), fmt.Sprintf("CASE kind %s ELSE %d END ASC, created_at ASC", strings.Join(orderParts, " "), priority)
}
