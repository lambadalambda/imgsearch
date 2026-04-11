package worker

import (
	"context"
	"database/sql"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
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
