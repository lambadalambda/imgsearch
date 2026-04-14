package worker

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
	"imgsearch/internal/embedder"
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

type fakeIndex struct {
	upserts []vectorindex.SearchHit
	err     error
}

type fakeAnnotator struct {
	annotation embedder.ImageAnnotation
	err        error
}

func (f *fakeAnnotator) AnnotateImage(context.Context, string) (embedder.ImageAnnotation, error) {
	if f.err != nil {
		return embedder.ImageAnnotation{}, f.err
	}
	return f.annotation, nil
}

func (f *fakeIndex) Upsert(_ context.Context, imageID int64, modelID int64, _ []float32) error {
	if f.err != nil {
		return f.err
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
		Index:         &fakeIndex{},
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

func TestProcessOneStoresGeneratedAnnotationsWhenAvailable(t *testing.T) {
	q, sqlDB := setupQueueTest(t)
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
}

func TestProcessOneStillCompletesWhenAnnotationGenerationFails(t *testing.T) {
	q, sqlDB := setupQueueTest(t)
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
	if err := sqlDB.QueryRow(`SELECT j.state, i.description, i.tags_json FROM index_jobs j JOIN images i ON i.id = j.image_id WHERE j.id = 1`).Scan(&state, &description, &tagsJSON); err != nil {
		t.Fatalf("load final state: %v", err)
	}
	if state != "done" {
		t.Fatalf("expected job state done, got %s", state)
	}
	if description != "" || tagsJSON != "[]" {
		t.Fatalf("expected annotations to remain empty, got description=%q tags_json=%q", description, tagsJSON)
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
