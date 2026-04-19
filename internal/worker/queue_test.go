package worker

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
	"imgsearch/internal/embedder"
	"imgsearch/internal/transcribe"
	"imgsearch/internal/vectorindex"
)

type fakeEmbedder struct {
	vec []float32
	err error
}

func (f *fakeEmbedder) EmbedImage(_ context.Context, _ string) ([]float32, error) {
	if f.err != nil {
		return nil, f.err
	}
	out := make([]float32, len(f.vec))
	copy(out, f.vec)
	return out, nil
}

func (f *fakeEmbedder) EmbedText(_ context.Context, _ string) ([]float32, error) {
	if f.err != nil {
		return nil, f.err
	}
	out := make([]float32, len(f.vec))
	copy(out, f.vec)
	return out, nil
}

type fakeBatchEmbedder struct {
	vec             []float32
	err             error
	batchErr        error
	embedImageCalls int
	embedBatchCalls int
	batchPaths      []string
}

func (f *fakeBatchEmbedder) EmbedImage(_ context.Context, _ string) ([]float32, error) {
	f.embedImageCalls++
	if f.err != nil {
		return nil, f.err
	}
	out := make([]float32, len(f.vec))
	copy(out, f.vec)
	return out, nil
}

func (f *fakeBatchEmbedder) EmbedImages(_ context.Context, paths []string) ([][]float32, error) {
	f.embedBatchCalls++
	f.batchPaths = append([]string(nil), paths...)
	if f.batchErr != nil {
		return nil, f.batchErr
	}
	out := make([][]float32, len(paths))
	for i := range paths {
		vec := make([]float32, len(f.vec))
		copy(vec, f.vec)
		out[i] = vec
	}
	return out, nil
}

type fakeIndex struct {
	db      *sql.DB
	upserts []vectorindex.SearchHit
	err     error
}

type fakeAnnotator struct {
	annotation      embedder.ImageAnnotation
	err             error
	lastOpts        embedder.ImageAnnotationOptions
	optsCalls       int
	videoAnnotation embedder.VideoAnnotation
	videoErr        error
	lastVideoInput  embedder.VideoAnnotationInput
	videoCalls      int
}

type fakeVideoTranscriber struct {
	transcript transcribe.Transcript
	err        error
}

func (f *fakeVideoTranscriber) TranscribeVideo(context.Context, string) (transcribe.Transcript, error) {
	if f.err != nil {
		return transcribe.Transcript{}, f.err
	}
	return f.transcript, nil
}

func (f *fakeAnnotator) AnnotateImage(context.Context, string) (embedder.ImageAnnotation, error) {
	if f.err != nil {
		return embedder.ImageAnnotation{}, f.err
	}
	return f.annotation, nil
}

func (f *fakeAnnotator) AnnotateImageWithOptions(_ context.Context, _ string, opts embedder.ImageAnnotationOptions) (embedder.ImageAnnotation, error) {
	f.optsCalls++
	f.lastOpts = opts
	if f.err != nil {
		return embedder.ImageAnnotation{}, f.err
	}
	return f.annotation, nil
}

func (f *fakeAnnotator) AnnotateVideo(_ context.Context, input embedder.VideoAnnotationInput) (embedder.VideoAnnotation, error) {
	f.videoCalls++
	f.lastVideoInput = input
	if f.videoErr != nil {
		return embedder.VideoAnnotation{}, f.videoErr
	}
	return f.videoAnnotation, nil
}

func (f *fakeIndex) Upsert(_ context.Context, imageID int64, modelID int64, vec []float32) error {
	if f.err != nil {
		return f.err
	}
	if f.db != nil {
		if _, err := f.db.Exec(`
INSERT INTO image_embeddings(image_id, model_id, dim, vector_blob)
VALUES (?, ?, ?, ?)
ON CONFLICT(image_id, model_id)
DO UPDATE SET
  dim = excluded.dim,
  vector_blob = excluded.vector_blob,
  updated_at = datetime('now')
`, imageID, modelID, len(vec), vectorindex.FloatsToBlob(vec)); err != nil {
			return err
		}
	}
	f.upserts = append(f.upserts, vectorindex.SearchHit{ImageID: imageID, ModelID: modelID})
	return nil
}

func (f *fakeIndex) Delete(context.Context, int64, int64) error { return nil }

func (f *fakeIndex) Search(context.Context, int64, []float32, int) ([]vectorindex.SearchHit, error) {
	return nil, nil
}

func (f *fakeIndex) SearchByImageID(context.Context, int64, int64, int) ([]vectorindex.SearchHit, error) {
	return nil, nil
}

func setupQueueTest(t *testing.T) (*Queue, *sql.DB) {
	t.Helper()

	sqlDB, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = sqlDB.Close() })

	if err := db.RunMigrations(context.Background(), sqlDB); err != nil {
		t.Fatalf("run migrations: %v", err)
	}

	modelID, err := db.EnsureEmbeddingModel(context.Background(), sqlDB, db.EmbeddingModelSpec{
		Name:       "test-model",
		Version:    "v1",
		Dimensions: 4,
		Metric:     "cosine",
		Normalized: true,
	})
	if err != nil {
		t.Fatalf("ensure model: %v", err)
	}

	dataDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dataDir, "images"), 0o755); err != nil {
		t.Fatalf("mkdir images dir: %v", err)
	}

	q := &Queue{
		DB:            sqlDB,
		DataDir:       dataDir,
		LeaseDuration: 30 * time.Second,
		Embedder:      &fakeEmbedder{vec: []float32{1, 2, 3, 4}},
		TextEmbedder:  &fakeEmbedder{vec: []float32{1, 2, 3, 4}},
		Index:         &fakeIndex{db: sqlDB},
	}

	// Ensure at least one fixture image and job helper data exist.
	if _, err := sqlDB.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES (1, 'abc', 'a.jpg', 'images/abc', 'image/jpeg', 10, 10)
`); err != nil {
		t.Fatalf("insert image: %v", err)
	}

	if err := os.WriteFile(filepath.Join(dataDir, "images", "abc"), []byte("test"), 0o644); err != nil {
		t.Fatalf("write image file: %v", err)
	}

	if _, err := sqlDB.Exec(`
INSERT INTO index_jobs(kind, image_id, model_id, state)
VALUES ('embed_image', 1, ?, 'pending')
`, modelID); err != nil {
		t.Fatalf("insert job: %v", err)
	}

	return q, sqlDB
}

func seedVideoTranscribeJob(t *testing.T, q *Queue, sqlDB *sql.DB, videoID int64, jobID int64) int64 {
	t.Helper()

	if _, err := sqlDB.Exec(`
INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count)
VALUES (?, ?, ?, ?, 'video/mp4', 1000, 640, 360, 1)
`, videoID, fmt.Sprintf("vid%d", videoID), fmt.Sprintf("clip-%d.mp4", videoID), fmt.Sprintf("videos/vid%d", videoID)); err != nil {
		t.Fatalf("insert video: %v", err)
	}
	if _, err := sqlDB.Exec(`
INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms)
VALUES (?, 1, 0, 0)
`, videoID); err != nil {
		t.Fatalf("insert video frame: %v", err)
	}
	if _, err := sqlDB.Exec(`DELETE FROM index_jobs WHERE id = 1`); err != nil {
		t.Fatalf("delete default job: %v", err)
	}

	var modelID int64
	if err := sqlDB.QueryRow(`SELECT id FROM embedding_models LIMIT 1`).Scan(&modelID); err != nil {
		t.Fatalf("select model id: %v", err)
	}

	if _, err := sqlDB.Exec(`
INSERT INTO index_jobs(id, kind, image_id, video_id, model_id, state)
VALUES (?, 'transcribe_video', NULL, ?, ?, 'pending')
`, jobID, videoID, modelID); err != nil {
		t.Fatalf("insert transcribe job: %v", err)
	}

	if err := os.MkdirAll(filepath.Join(q.DataDir, "videos"), 0o755); err != nil {
		t.Fatalf("mkdir videos dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(q.DataDir, "videos", fmt.Sprintf("vid%d", videoID)), []byte("video"), 0o644); err != nil {
		t.Fatalf("write video file: %v", err)
	}

	return modelID
}

func TestProcessOneSuccessStoresEmbeddingAndMarksDone(t *testing.T) {
	q, sqlDB := setupQueueTest(t)

	processed, err := q.ProcessOne(context.Background(), "worker-1")
	if err != nil {
		t.Fatalf("process one: %v", err)
	}
	if !processed {
		t.Fatal("expected processed=true")
	}

	var state string
	if err := sqlDB.QueryRow(`SELECT state FROM index_jobs WHERE id = 1`).Scan(&state); err != nil {
		t.Fatalf("select job state: %v", err)
	}
	if state != "done" {
		t.Fatalf("expected job state done, got %s", state)
	}

	var dim int
	if err := sqlDB.QueryRow(`SELECT dim FROM image_embeddings WHERE image_id = 1`).Scan(&dim); err != nil {
		t.Fatalf("select embedding dim: %v", err)
	}
	if dim != 4 {
		t.Fatalf("expected dim 4, got %d", dim)
	}
}

func TestProcessOneTranscribeVideoStoresTranscriptAndEmbedding(t *testing.T) {
	q, sqlDB := setupQueueTest(t)
	q.Transcriber = &fakeVideoTranscriber{transcript: transcribe.Transcript{Text: "tis better to remain silent"}}
	q.TextEmbedder = &fakeEmbedder{vec: []float32{9, 8, 7, 6}}

	if _, err := sqlDB.Exec(`
INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count)
VALUES (3, 'vid3', 'clip.mp4', 'videos/vid3', 'video/mp4', 1000, 640, 360, 1)
`); err != nil {
		t.Fatalf("insert video: %v", err)
	}
	if _, err := sqlDB.Exec(`
INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms)
VALUES (3, 1, 0, 0)
`); err != nil {
		t.Fatalf("insert video frame: %v", err)
	}
	if _, err := sqlDB.Exec(`DELETE FROM index_jobs WHERE id = 1`); err != nil {
		t.Fatalf("delete default job: %v", err)
	}
	var modelID int64
	if err := sqlDB.QueryRow(`SELECT id FROM embedding_models LIMIT 1`).Scan(&modelID); err != nil {
		t.Fatalf("select model id: %v", err)
	}
	if _, err := sqlDB.Exec(`
INSERT INTO index_jobs(id, kind, image_id, video_id, model_id, state)
VALUES (10, 'transcribe_video', NULL, 3, ?, 'pending')
`, modelID); err != nil {
		t.Fatalf("insert transcribe job: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(q.DataDir, "videos"), 0o755); err != nil {
		t.Fatalf("mkdir videos dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(q.DataDir, "videos", "vid3"), []byte("video"), 0o644); err != nil {
		t.Fatalf("write video file: %v", err)
	}

	processed, err := q.ProcessOne(context.Background(), "worker-1")
	if err != nil {
		t.Fatalf("process one: %v", err)
	}
	if !processed {
		t.Fatal("expected processed=true")
	}

	var transcriptText string
	if err := sqlDB.QueryRow(`SELECT transcript_text FROM videos WHERE id = 3`).Scan(&transcriptText); err != nil {
		t.Fatalf("select transcript: %v", err)
	}
	if transcriptText != "tis better to remain silent" {
		t.Fatalf("unexpected transcript text: %q", transcriptText)
	}

	var dim int
	if err := sqlDB.QueryRow(`SELECT dim FROM video_transcript_embeddings WHERE video_id = 3 AND model_id = ?`, modelID).Scan(&dim); err != nil {
		t.Fatalf("select transcript embedding dim: %v", err)
	}
	if dim != 4 {
		t.Fatalf("expected transcript dim 4, got %d", dim)
	}
}

func TestProcessOneTranscribeVideoWithEmptyTranscriptMarksDoneWithoutEmbedding(t *testing.T) {
	q, sqlDB := setupQueueTest(t)
	q.Transcriber = &fakeVideoTranscriber{transcript: transcribe.Transcript{Text: "   "}}
	q.TextEmbedder = &fakeEmbedder{err: errors.New("text embedder should not be called for empty transcript")}

	modelID := seedVideoTranscribeJob(t, q, sqlDB, 6, 16)

	processed, err := q.ProcessOne(context.Background(), "worker-1")
	if err != nil {
		t.Fatalf("process one: %v", err)
	}
	if !processed {
		t.Fatal("expected processed=true")
	}

	var transcriptText string
	if err := sqlDB.QueryRow(`SELECT transcript_text FROM videos WHERE id = 6`).Scan(&transcriptText); err != nil {
		t.Fatalf("select transcript: %v", err)
	}
	if transcriptText != "" {
		t.Fatalf("expected empty transcript text, got %q", transcriptText)
	}

	var jobState string
	if err := sqlDB.QueryRow(`SELECT state FROM index_jobs WHERE id = 16`).Scan(&jobState); err != nil {
		t.Fatalf("select transcribe job state: %v", err)
	}
	if jobState != "done" {
		t.Fatalf("expected transcribe job done, got %s", jobState)
	}

	var transcriptEmbeddings int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM video_transcript_embeddings WHERE video_id = 6 AND model_id = ?`, modelID).Scan(&transcriptEmbeddings); err != nil {
		t.Fatalf("count transcript embeddings: %v", err)
	}
	if transcriptEmbeddings != 0 {
		t.Fatalf("expected no transcript embeddings for empty transcript, got %d", transcriptEmbeddings)
	}
}

func TestProcessOneTranscribeVideoWithoutAudioMarksDoneWithoutEmbedding(t *testing.T) {
	q, sqlDB := setupQueueTest(t)
	q.Transcriber = &fakeVideoTranscriber{err: transcribe.ErrNoAudio}
	q.TextEmbedder = &fakeEmbedder{err: errors.New("text embedder should not be called when video has no audio")}

	modelID := seedVideoTranscribeJob(t, q, sqlDB, 7, 17)

	processed, err := q.ProcessOne(context.Background(), "worker-1")
	if err != nil {
		t.Fatalf("process one: %v", err)
	}
	if !processed {
		t.Fatal("expected processed=true")
	}

	var transcriptText string
	if err := sqlDB.QueryRow(`SELECT transcript_text FROM videos WHERE id = 7`).Scan(&transcriptText); err != nil {
		t.Fatalf("select transcript: %v", err)
	}
	if transcriptText != "" {
		t.Fatalf("expected empty transcript text for no-audio video, got %q", transcriptText)
	}

	var jobState string
	if err := sqlDB.QueryRow(`SELECT state FROM index_jobs WHERE id = 17`).Scan(&jobState); err != nil {
		t.Fatalf("select transcribe job state: %v", err)
	}
	if jobState != "done" {
		t.Fatalf("expected transcribe job done for no-audio video, got %s", jobState)
	}

	var transcriptEmbeddings int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM video_transcript_embeddings WHERE video_id = 7 AND model_id = ?`, modelID).Scan(&transcriptEmbeddings); err != nil {
		t.Fatalf("count transcript embeddings: %v", err)
	}
	if transcriptEmbeddings != 0 {
		t.Fatalf("expected no transcript embeddings for no-audio video, got %d", transcriptEmbeddings)
	}
}

func TestProcessOneDoesNotPersistEmbeddingWhenIndexUpsertFails(t *testing.T) {
	q, sqlDB := setupQueueTest(t)
	q.Index = &fakeIndex{db: sqlDB, err: errors.New("index unavailable")}

	processed, err := q.ProcessOne(context.Background(), "worker-1")
	if err == nil {
		t.Fatal("expected process error")
	}
	if !processed {
		t.Fatal("expected processed=true")
	}

	var state string
	if err := sqlDB.QueryRow(`SELECT state FROM index_jobs WHERE id = 1`).Scan(&state); err != nil {
		t.Fatalf("select job state: %v", err)
	}
	if state != "pending" {
		t.Fatalf("expected pending state after index failure, got %s", state)
	}

	var count int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM image_embeddings WHERE image_id = 1`).Scan(&count); err != nil {
		t.Fatalf("count embeddings: %v", err)
	}
	if count != 0 {
		t.Fatalf("expected no persisted embedding after index failure, got %d rows", count)
	}
}

func TestProcessOneEmbedJobLeavesAnnotationsUntouched(t *testing.T) {
	q, sqlDB := setupQueueTest(t)
	if _, err := sqlDB.Exec(`
INSERT INTO index_jobs(id, kind, image_id, model_id, state)
VALUES (2, 'annotate_image', 1, 1, 'pending')
`); err != nil {
		t.Fatalf("insert annotate job: %v", err)
	}
	q.Annotator = &fakeAnnotator{annotation: embedder.ImageAnnotation{
		Description: "A calm test image.",
		Tags:        []string{"test", "sample", "nsfw"},
	}}

	processed, err := q.ProcessOne(context.Background(), "worker-1")
	if err != nil {
		t.Fatalf("process one: %v", err)
	}
	if !processed {
		t.Fatal("expected processed=true")
	}

	var description string
	var tagsJSON string
	if err := sqlDB.QueryRow(`SELECT description, tags_json FROM images WHERE id = 1`).Scan(&description, &tagsJSON); err != nil {
		t.Fatalf("load stored annotations: %v", err)
	}
	if description != "" || tagsJSON != "[]" {
		t.Fatalf("expected embed job to leave annotations untouched, got description=%q tags_json=%q", description, tagsJSON)
	}

	var embedState string
	if err := sqlDB.QueryRow(`SELECT state FROM index_jobs WHERE id = 1`).Scan(&embedState); err != nil {
		t.Fatalf("load embed job state: %v", err)
	}
	if embedState != "done" {
		t.Fatalf("expected embed job done, got %s", embedState)
	}

	var annotateState string
	if err := sqlDB.QueryRow(`SELECT state FROM index_jobs WHERE id = 2`).Scan(&annotateState); err != nil {
		t.Fatalf("load annotate job state: %v", err)
	}
	if annotateState != "pending" {
		t.Fatalf("expected annotate job to remain pending, got %s", annotateState)
	}
}

func TestProcessOneProcessesAnnotateJobAndStoresGeneratedAnnotationsWhenAvailable(t *testing.T) {
	q, sqlDB := setupQueueTest(t)
	if _, err := sqlDB.Exec(`UPDATE index_jobs SET state = 'done' WHERE id = 1`); err != nil {
		t.Fatalf("mark embed job done: %v", err)
	}
	if _, err := sqlDB.Exec(`
INSERT INTO index_jobs(id, kind, image_id, model_id, state)
VALUES (2, 'annotate_image', 1, 1, 'pending')
`); err != nil {
		t.Fatalf("insert annotate job: %v", err)
	}
	annotator := &fakeAnnotator{annotation: embedder.ImageAnnotation{
		Description: "A calm test image.",
		Tags:        []string{"test", "sample", "nsfw"},
	}}
	q.Annotator = annotator

	processed, err := q.ProcessOne(context.Background(), "worker-1")
	if err != nil {
		t.Fatalf("process one: %v", err)
	}
	if !processed {
		t.Fatal("expected processed=true")
	}

	var description string
	var tagsJSON string
	if err := sqlDB.QueryRow(`SELECT description, tags_json FROM images WHERE id = 1`).Scan(&description, &tagsJSON); err != nil {
		t.Fatalf("load stored annotations: %v", err)
	}
	if description != "A calm test image." {
		t.Fatalf("unexpected description: %q", description)
	}
	var tags []string
	if err := json.Unmarshal([]byte(tagsJSON), &tags); err != nil {
		t.Fatalf("decode stored tags: %v", err)
	}
	if len(tags) != 3 || tags[2] != "nsfw" {
		t.Fatalf("unexpected tags: %v", tags)
	}

	var annotateState string
	if err := sqlDB.QueryRow(`SELECT state FROM index_jobs WHERE id = 2`).Scan(&annotateState); err != nil {
		t.Fatalf("load annotate job state: %v", err)
	}
	if annotateState != "done" {
		t.Fatalf("expected annotate job done, got %s", annotateState)
	}
	if annotator.optsCalls != 1 {
		t.Fatalf("expected annotate_image to use options-aware call once, got %d", annotator.optsCalls)
	}
	if annotator.lastOpts.OriginalName != "a.jpg" {
		t.Fatalf("expected annotate options to include original filename, got %q", annotator.lastOpts.OriginalName)
	}
}

func TestProcessOneProcessesAnnotateVideoJobAndStoresVideoAnnotations(t *testing.T) {
	q, sqlDB := setupQueueTest(t)
	if _, err := sqlDB.Exec(`DELETE FROM index_jobs WHERE id = 1`); err != nil {
		t.Fatalf("delete default embed job: %v", err)
	}
	if _, err := sqlDB.Exec(`
UPDATE images
SET description = 'A singer on stage under bright lights.',
    tags_json = '["frame","stage"]'
WHERE id = 1
`); err != nil {
		t.Fatalf("seed frame annotations: %v", err)
	}
	if _, err := sqlDB.Exec(`
INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count, transcript_text)
VALUES (9, 'vid9', 'concert_clip.mp4', 'videos/vid9', 'video/mp4', 9000, 1280, 720, 1, 'crowd cheering loudly')
`); err != nil {
		t.Fatalf("insert video: %v", err)
	}
	if _, err := sqlDB.Exec(`
INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms)
VALUES (9, 1, 0, 1000)
`); err != nil {
		t.Fatalf("insert video frame: %v", err)
	}
	if _, err := sqlDB.Exec(`
INSERT INTO index_jobs(id, kind, image_id, video_id, model_id, state)
VALUES (30, 'annotate_video', NULL, 9, 1, 'pending')
`); err != nil {
		t.Fatalf("insert annotate_video job: %v", err)
	}

	annotator := &fakeAnnotator{videoAnnotation: embedder.VideoAnnotation{
		Description: "A live concert performance with energetic stage lighting and crowd movement.",
		Tags:        []string{"concert", "music", "crowd"},
		IsNSFW:      false,
	}}
	q.Annotator = annotator

	processed, err := q.ProcessOne(context.Background(), "worker-1")
	if err != nil {
		t.Fatalf("process one: %v", err)
	}
	if !processed {
		t.Fatal("expected processed=true")
	}

	if annotator.videoCalls != 1 {
		t.Fatalf("expected one video annotation call, got %d", annotator.videoCalls)
	}
	if annotator.lastVideoInput.OriginalName != "concert_clip.mp4" {
		t.Fatalf("expected video input original name, got %q", annotator.lastVideoInput.OriginalName)
	}
	if annotator.lastVideoInput.RepresentativeFramePath != filepath.Join(q.DataDir, filepath.FromSlash("images/abc")) {
		t.Fatalf("expected representative frame path %q, got %q", filepath.Join(q.DataDir, filepath.FromSlash("images/abc")), annotator.lastVideoInput.RepresentativeFramePath)
	}
	if len(annotator.lastVideoInput.Frames) != 1 {
		t.Fatalf("expected one frame annotation in video input, got %d", len(annotator.lastVideoInput.Frames))
	}

	var storedDescription string
	var storedTagsJSON string
	if err := sqlDB.QueryRow(`SELECT description, tags_json FROM videos WHERE id = 9`).Scan(&storedDescription, &storedTagsJSON); err != nil {
		t.Fatalf("load stored video annotations: %v", err)
	}
	if storedDescription != "A live concert performance with energetic stage lighting and crowd movement." {
		t.Fatalf("unexpected stored video description: %q", storedDescription)
	}
	var storedTags []string
	if err := json.Unmarshal([]byte(storedTagsJSON), &storedTags); err != nil {
		t.Fatalf("decode stored video tags: %v", err)
	}
	if len(storedTags) != 3 || storedTags[0] != "concert" {
		t.Fatalf("unexpected stored video tags: %v", storedTags)
	}

	var jobState string
	if err := sqlDB.QueryRow(`SELECT state FROM index_jobs WHERE id = 30`).Scan(&jobState); err != nil {
		t.Fatalf("load annotate_video state: %v", err)
	}
	if jobState != "done" {
		t.Fatalf("expected annotate_video job done, got %s", jobState)
	}
}

func TestProcessOneAnnotateVideoWithMissingFrameAnnotationsSingleConn(t *testing.T) {
	q, sqlDB := setupQueueTest(t)
	sqlDB.SetMaxOpenConns(1)
	sqlDB.SetMaxIdleConns(1)

	if _, err := sqlDB.Exec(`DELETE FROM index_jobs WHERE id = 1`); err != nil {
		t.Fatalf("delete default embed job: %v", err)
	}
	if _, err := sqlDB.Exec(`
INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count, transcript_text)
VALUES (10, 'vid10', 'clip.mp4', 'videos/vid10', 'video/mp4', 5000, 1280, 720, 1, 'hello crowd')
`); err != nil {
		t.Fatalf("insert video: %v", err)
	}
	if _, err := sqlDB.Exec(`
INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms)
VALUES (10, 1, 0, 1000)
`); err != nil {
		t.Fatalf("insert video frame: %v", err)
	}
	if _, err := sqlDB.Exec(`
INSERT INTO index_jobs(id, kind, image_id, video_id, model_id, state)
VALUES (31, 'annotate_video', NULL, 10, 1, 'pending')
`); err != nil {
		t.Fatalf("insert annotate_video job: %v", err)
	}

	annotator := &fakeAnnotator{
		annotation: embedder.ImageAnnotation{
			Description: "A frame showing a cheering crowd.",
			Tags:        []string{"crowd", "stage"},
		},
		videoAnnotation: embedder.VideoAnnotation{
			Description: "A concert clip with a lively crowd and stage lights.",
			Tags:        []string{"concert", "crowd", "stage"},
		},
	}
	q.Annotator = annotator

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	processed, err := q.ProcessOne(ctx, "worker-1")
	if err != nil {
		t.Fatalf("process one: %v", err)
	}
	if !processed {
		t.Fatal("expected processed=true")
	}
	if annotator.optsCalls != 1 {
		t.Fatalf("expected one frame annotate_image call, got %d", annotator.optsCalls)
	}
	if annotator.videoCalls != 1 {
		t.Fatalf("expected one video annotation call, got %d", annotator.videoCalls)
	}

	var frameDescription string
	var frameTagsJSON string
	if err := sqlDB.QueryRow(`SELECT description, tags_json FROM images WHERE id = 1`).Scan(&frameDescription, &frameTagsJSON); err != nil {
		t.Fatalf("load frame annotations: %v", err)
	}
	if frameDescription == "" {
		t.Fatalf("expected frame description to be stored")
	}

	var jobState string
	if err := sqlDB.QueryRow(`SELECT state FROM index_jobs WHERE id = 31`).Scan(&jobState); err != nil {
		t.Fatalf("load annotate_video state: %v", err)
	}
	if jobState != "done" {
		t.Fatalf("expected annotate_video job done, got %s", jobState)
	}
}

func TestProcessOneRetriesAnnotateJobOnFailure(t *testing.T) {
	q, sqlDB := setupQueueTest(t)
	if _, err := sqlDB.Exec(`UPDATE index_jobs SET state = 'done' WHERE id = 1`); err != nil {
		t.Fatalf("mark embed job done: %v", err)
	}
	if _, err := sqlDB.Exec(`
INSERT INTO index_jobs(id, kind, image_id, model_id, state)
VALUES (2, 'annotate_image', 1, 1, 'pending')
`); err != nil {
		t.Fatalf("insert annotate job: %v", err)
	}
	q.Annotator = &fakeAnnotator{err: errors.New("annotate failed")}

	processed, err := q.ProcessOne(context.Background(), "worker-1")
	if err != nil {
		t.Fatalf("process one: %v", err)
	}
	if !processed {
		t.Fatal("expected processed=true")
	}

	var state string
	var description string
	var tagsJSON string
	if err := sqlDB.QueryRow(`SELECT j.state, i.description, i.tags_json FROM index_jobs j JOIN images i ON i.id = j.image_id WHERE j.id = 2`).Scan(&state, &description, &tagsJSON); err != nil {
		t.Fatalf("load final state: %v", err)
	}
	if state != "pending" {
		t.Fatalf("expected annotate job state pending for retry, got %s", state)
	}
	if description != "" || tagsJSON != "[]" {
		t.Fatalf("expected annotations to remain empty, got description=%q tags_json=%q", description, tagsJSON)
	}
}

func TestProcessOneSkipsAnnotateJobsWhenNoAnnotatorAvailable(t *testing.T) {
	q, sqlDB := setupQueueTest(t)
	if _, err := sqlDB.Exec(`UPDATE index_jobs SET state = 'done' WHERE id = 1`); err != nil {
		t.Fatalf("mark embed job done: %v", err)
	}
	if _, err := sqlDB.Exec(`
INSERT INTO index_jobs(id, kind, image_id, model_id, state)
VALUES (2, 'annotate_image', 1, 1, 'pending')
`); err != nil {
		t.Fatalf("insert annotate job: %v", err)
	}

	processed, err := q.ProcessOne(context.Background(), "worker-1")
	if err != nil {
		t.Fatalf("process one: %v", err)
	}
	if processed {
		t.Fatal("expected processed=false when only annotate jobs remain and annotator is unavailable")
	}
}

func TestProcessOneRetriesThenFailsAfterMaxAttempts(t *testing.T) {
	q, sqlDB := setupQueueTest(t)
	q.Embedder = &fakeEmbedder{err: errors.New("embed failed")}

	for i := 1; i <= 3; i++ {
		processed, err := q.ProcessOne(context.Background(), "worker-1")
		if err != nil {
			t.Fatalf("attempt %d process one: %v", i, err)
		}
		if !processed {
			t.Fatalf("attempt %d expected processed=true", i)
		}
	}

	var state string
	var attempts int
	if err := sqlDB.QueryRow(`SELECT state, attempts FROM index_jobs WHERE id = 1`).Scan(&state, &attempts); err != nil {
		t.Fatalf("select job final state: %v", err)
	}
	if state != "failed" {
		t.Fatalf("expected failed state, got %s", state)
	}
	if attempts != 3 {
		t.Fatalf("expected attempts=3, got %d", attempts)
	}
}

func TestProcessOneReturnsFalseWhenNoEligibleJobs(t *testing.T) {
	q, _ := setupQueueTest(t)

	// First run consumes the only pending job.
	_, _ = q.ProcessOne(context.Background(), "worker-1")

	processed, err := q.ProcessOne(context.Background(), "worker-1")
	if err != nil {
		t.Fatalf("process one: %v", err)
	}
	if processed {
		t.Fatal("expected processed=false when queue is empty")
	}
}

func TestProcessOneReclaimsExpiredLease(t *testing.T) {
	q, sqlDB := setupQueueTest(t)

	if _, err := sqlDB.Exec(`
UPDATE index_jobs
SET state = 'leased', leased_until = datetime('now', '-1 minute')
WHERE id = 1
`); err != nil {
		t.Fatalf("update job lease: %v", err)
	}

	processed, err := q.ProcessOne(context.Background(), "worker-1")
	if err != nil {
		t.Fatalf("process one: %v", err)
	}
	if !processed {
		t.Fatal("expected lease reclaim to process a job")
	}

	var state string
	if err := sqlDB.QueryRow(`SELECT state FROM index_jobs WHERE id = 1`).Scan(&state); err != nil {
		t.Fatalf("select state: %v", err)
	}
	if state != "done" {
		t.Fatalf("expected done after reclaim, got %s", state)
	}
}

func setupMultiImageQueue(t *testing.T, imageCount int) (*Queue, *sql.DB) {
	t.Helper()

	sqlDB, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = sqlDB.Close() })

	if err := db.RunMigrations(context.Background(), sqlDB); err != nil {
		t.Fatalf("run migrations: %v", err)
	}

	modelID, err := db.EnsureEmbeddingModel(context.Background(), sqlDB, db.EmbeddingModelSpec{
		Name:       "test-model",
		Version:    "v1",
		Dimensions: 4,
		Metric:     "cosine",
		Normalized: true,
	})
	if err != nil {
		t.Fatalf("ensure model: %v", err)
	}

	dataDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dataDir, "images"), 0o755); err != nil {
		t.Fatalf("mkdir images dir: %v", err)
	}

	for i := 1; i <= imageCount; i++ {
		sha := fmt.Sprintf("sha%d", i)
		name := fmt.Sprintf("img%d.jpg", i)
		storagePath := fmt.Sprintf("images/%s", sha)
		if _, err := sqlDB.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES (?, ?, ?, ?, 'image/jpeg', 10, 10)
`, int64(i), sha, name, storagePath); err != nil {
			t.Fatalf("insert image %d: %v", i, err)
		}
		if err := os.WriteFile(filepath.Join(dataDir, "images", sha), []byte("test"), 0o644); err != nil {
			t.Fatalf("write image file %d: %v", i, err)
		}
		if _, err := sqlDB.Exec(`
INSERT INTO index_jobs(kind, image_id, model_id, state)
VALUES ('embed_image', ?, ?, 'pending')
`, int64(i), modelID); err != nil {
			t.Fatalf("insert job %d: %v", i, err)
		}
	}

	q := &Queue{
		DB:            sqlDB,
		DataDir:       dataDir,
		LeaseDuration: 30 * time.Second,
		Embedder:      &fakeEmbedder{vec: []float32{1, 2, 3, 4}},
		Index:         &fakeIndex{db: sqlDB},
	}

	return q, sqlDB
}

func TestClaimBatchClaimsUpToLimitJobsOfSameKind(t *testing.T) {
	q, _ := setupMultiImageQueue(t, 5)

	jobs, err := q.claimBatch(context.Background(), "worker-1", 3)
	if err != nil {
		t.Fatalf("claim batch: %v", err)
	}
	if len(jobs) != 3 {
		t.Fatalf("expected 3 jobs, got %d", len(jobs))
	}
	for _, j := range jobs {
		if j.Kind != "embed_image" {
			t.Fatalf("expected embed_image kind, got %s", j.Kind)
		}
		if j.Attempts != 1 {
			t.Fatalf("expected attempts=1, got %d", j.Attempts)
		}
	}
}

func TestClaimBatchReturnsFewerWhenNotEnoughEligible(t *testing.T) {
	q, _ := setupMultiImageQueue(t, 2)

	jobs, err := q.claimBatch(context.Background(), "worker-1", 5)
	if err != nil {
		t.Fatalf("claim batch: %v", err)
	}
	if len(jobs) != 2 {
		t.Fatalf("expected 2 jobs, got %d", len(jobs))
	}
}

func TestClaimBatchReturnsEmptyWhenNoEligibleJobs(t *testing.T) {
	q, _ := setupMultiImageQueue(t, 2)

	_, _ = q.claimBatch(context.Background(), "worker-1", 10)

	jobs, err := q.claimBatch(context.Background(), "worker-1", 5)
	if err != nil {
		t.Fatalf("claim batch: %v", err)
	}
	if len(jobs) != 0 {
		t.Fatalf("expected 0 jobs, got %d", len(jobs))
	}
}

func TestClaimBatchAllLeasedWithSameOwner(t *testing.T) {
	q, sqlDB := setupMultiImageQueue(t, 3)

	jobs, err := q.claimBatch(context.Background(), "batch-worker", 3)
	if err != nil {
		t.Fatalf("claim batch: %v", err)
	}
	if len(jobs) != 3 {
		t.Fatalf("expected 3 jobs, got %d", len(jobs))
	}

	for _, j := range jobs {
		var owner string
		var state string
		if err := sqlDB.QueryRow(`SELECT lease_owner, state FROM index_jobs WHERE id = ?`, j.ID).Scan(&owner, &state); err != nil {
			t.Fatalf("select job %d: %v", j.ID, err)
		}
		if owner != "batch-worker" {
			t.Fatalf("expected owner batch-worker, got %s", owner)
		}
		if state != "leased" {
			t.Fatalf("expected state leased, got %s", state)
		}
	}
}

func TestClaimBatchSkipsExpiredLeasesAndPendingOnly(t *testing.T) {
	q, sqlDB := setupMultiImageQueue(t, 4)

	if _, err := sqlDB.Exec(`UPDATE index_jobs SET state = 'leased', leased_until = datetime('now', '-1 minute') WHERE image_id = 1`); err != nil {
		t.Fatalf("expire lease: %v", err)
	}
	if _, err := sqlDB.Exec(`UPDATE index_jobs SET state = 'done' WHERE image_id = 2`); err != nil {
		t.Fatalf("mark done: %v", err)
	}

	jobs, err := q.claimBatch(context.Background(), "worker-1", 4)
	if err != nil {
		t.Fatalf("claim batch: %v", err)
	}
	if len(jobs) != 3 {
		t.Fatalf("expected 3 jobs (1 expired lease + 2 pending, skipping 1 done), got %d", len(jobs))
	}
}

func TestClaimBatchWithAnnotatorClaimsEmbedFirst(t *testing.T) {
	q, sqlDB := setupMultiImageQueue(t, 3)
	if _, err := sqlDB.Exec(`UPDATE index_jobs SET state = 'done' WHERE image_id = 1`); err != nil {
		t.Fatalf("mark done: %v", err)
	}
	if _, err := sqlDB.Exec(`
INSERT INTO index_jobs(kind, image_id, model_id, state)
VALUES ('annotate_image', 1, 1, 'pending')
`); err != nil {
		t.Fatalf("insert annotate job: %v", err)
	}
	q.Annotator = &fakeAnnotator{annotation: embedder.ImageAnnotation{Description: "desc", Tags: []string{"t"}}}

	jobs, err := q.claimBatch(context.Background(), "worker-1", 5)
	if err != nil {
		t.Fatalf("claim batch: %v", err)
	}
	if len(jobs) < 2 {
		t.Fatalf("expected at least 2 jobs, got %d", len(jobs))
	}
	for _, j := range jobs {
		if j.Kind != "embed_image" {
			t.Fatalf("expected all batched jobs to be embed_image, got %s", j.Kind)
		}
	}
}

func TestProcessBatchProcessesMultipleJobs(t *testing.T) {
	q, sqlDB := setupMultiImageQueue(t, 5)

	count, err := q.ProcessBatch(context.Background(), "worker-1", 3)
	if err != nil {
		t.Fatalf("process batch: %v", err)
	}
	if count != 3 {
		t.Fatalf("expected 3 processed, got %d", count)
	}

	var doneCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE state = 'done'`).Scan(&doneCount); err != nil {
		t.Fatalf("count done: %v", err)
	}
	if doneCount != 3 {
		t.Fatalf("expected 3 done jobs, got %d", doneCount)
	}

	var pendingCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE state = 'pending'`).Scan(&pendingCount); err != nil {
		t.Fatalf("count pending: %v", err)
	}
	if pendingCount != 2 {
		t.Fatalf("expected 2 pending jobs, got %d", pendingCount)
	}
}

func TestProcessBatchReturnsZeroWhenEmpty(t *testing.T) {
	q, _ := setupMultiImageQueue(t, 0)

	count, err := q.ProcessBatch(context.Background(), "worker-1", 5)
	if err != nil {
		t.Fatalf("process batch: %v", err)
	}
	if count != 0 {
		t.Fatalf("expected 0 processed, got %d", count)
	}
}

func TestProcessBatchContinuesOnSingleJobFailure(t *testing.T) {
	q, sqlDB := setupMultiImageQueue(t, 3)

	if _, err := sqlDB.Exec(`DELETE FROM image_embeddings WHERE image_id = 2`); err != nil {
		t.Fatalf("cleanup: %v", err)
	}
	q.Index = &failingAtIndex{failAtImageID: 2, db: sqlDB}

	count, err := q.ProcessBatch(context.Background(), "worker-1", 3)
	if err != nil {
		t.Fatalf("process batch: %v", err)
	}
	if count != 2 {
		t.Fatalf("expected 2 processed (skipping 1 failure), got %d", count)
	}

	var doneCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE state = 'done'`).Scan(&doneCount); err != nil {
		t.Fatalf("count done: %v", err)
	}
	if doneCount != 2 {
		t.Fatalf("expected 2 done jobs, got %d", doneCount)
	}

	var pendingCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE state = 'pending'`).Scan(&pendingCount); err != nil {
		t.Fatalf("count pending: %v", err)
	}
	if pendingCount != 1 {
		t.Fatalf("expected 1 pending job (retried), got %d", pendingCount)
	}
}

func TestProcessBatchUsesEmbedImagesWhenAvailable(t *testing.T) {
	q, sqlDB := setupMultiImageQueue(t, 3)
	batch := &fakeBatchEmbedder{vec: []float32{1, 2, 3, 4}}
	q.Embedder = batch

	count, err := q.ProcessBatch(context.Background(), "worker-1", 3)
	if err != nil {
		t.Fatalf("process batch: %v", err)
	}
	if count != 3 {
		t.Fatalf("expected 3 processed jobs, got %d", count)
	}
	if batch.embedBatchCalls != 1 {
		t.Fatalf("expected 1 EmbedImages call, got %d", batch.embedBatchCalls)
	}
	if batch.embedImageCalls != 0 {
		t.Fatalf("expected 0 EmbedImage calls, got %d", batch.embedImageCalls)
	}
	if len(batch.batchPaths) != 3 {
		t.Fatalf("expected 3 batch paths, got %d", len(batch.batchPaths))
	}

	var doneCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE state = 'done'`).Scan(&doneCount); err != nil {
		t.Fatalf("count done: %v", err)
	}
	if doneCount != 3 {
		t.Fatalf("expected 3 done jobs, got %d", doneCount)
	}
}

func TestProcessBatchRetriesAllJobsWhenBatchEmbedFails(t *testing.T) {
	q, sqlDB := setupMultiImageQueue(t, 3)
	q.Embedder = &fakeBatchEmbedder{vec: []float32{1, 2, 3, 4}, batchErr: errors.New("batch failed")}

	count, err := q.ProcessBatch(context.Background(), "worker-1", 3)
	if err != nil {
		t.Fatalf("process batch: %v", err)
	}
	if count != 0 {
		t.Fatalf("expected 0 processed jobs, got %d", count)
	}

	var pendingCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE state = 'pending'`).Scan(&pendingCount); err != nil {
		t.Fatalf("count pending: %v", err)
	}
	if pendingCount != 3 {
		t.Fatalf("expected 3 pending jobs after retry, got %d", pendingCount)
	}
}

type failingAtIndex struct {
	failAtImageID int64
	db            *sql.DB
	upserts       int
}

func (f *failingAtIndex) Upsert(_ context.Context, imageID int64, modelID int64, vec []float32) error {
	if imageID == f.failAtImageID {
		return errors.New("index unavailable")
	}
	if f.db != nil {
		if _, err := f.db.Exec(`
INSERT INTO image_embeddings(image_id, model_id, dim, vector_blob)
VALUES (?, ?, ?, ?)
ON CONFLICT(image_id, model_id)
DO UPDATE SET
  dim = excluded.dim,
  vector_blob = excluded.vector_blob,
  updated_at = datetime('now')
`, imageID, modelID, len(vec), vectorindex.FloatsToBlob(vec)); err != nil {
			return err
		}
	}
	f.upserts++
	return nil
}

func (f *failingAtIndex) Delete(context.Context, int64, int64) error { return nil }

func (f *failingAtIndex) Search(context.Context, int64, []float32, int) ([]vectorindex.SearchHit, error) {
	return nil, nil
}

func (f *failingAtIndex) SearchByImageID(context.Context, int64, int64, int) ([]vectorindex.SearchHit, error) {
	return nil, nil
}

func TestProcessOneSetsRunAfterWhenRetryDelayConfigured(t *testing.T) {
	q, sqlDB := setupQueueTest(t)
	q.Embedder = &fakeEmbedder{err: errors.New("sidecar unavailable")}
	q.RetryBaseDelay = 2 * time.Second

	processed, err := q.ProcessOne(context.Background(), "worker-1")
	if err != nil {
		t.Fatalf("process one: %v", err)
	}
	if !processed {
		t.Fatal("expected processed=true")
	}

	var state string
	var runAfter sql.NullString
	if err := sqlDB.QueryRow(`SELECT state, run_after FROM index_jobs WHERE id = 1`).Scan(&state, &runAfter); err != nil {
		t.Fatalf("select job retry data: %v", err)
	}
	if state != "pending" {
		t.Fatalf("expected pending state for retry, got %s", state)
	}
	if !runAfter.Valid || runAfter.String == "" {
		t.Fatal("expected run_after to be set for delayed retry")
	}

	var future int
	if err := sqlDB.QueryRow(`
SELECT CASE WHEN run_after > datetime('now') THEN 1 ELSE 0 END
FROM index_jobs
WHERE id = 1
`).Scan(&future); err != nil {
		t.Fatalf("compare run_after with now: %v", err)
	}
	if future != 1 {
		t.Fatalf("expected run_after in the future, got run_after=%q", runAfter.String)
	}
}
