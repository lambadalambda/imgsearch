package db

import (
	"context"
	"database/sql"
	"strings"
	"testing"

	_ "github.com/mattn/go-sqlite3"
)

func openModelDB(t *testing.T) *sql.DB {
	t.Helper()

	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	if err := RunMigrations(context.Background(), db); err != nil {
		t.Fatalf("run migrations: %v", err)
	}
	return db
}

func TestEnsureEmbeddingModelCreatesThenReturnsExisting(t *testing.T) {
	db := openModelDB(t)
	ctx := context.Background()

	firstID, err := EnsureEmbeddingModel(ctx, db, EmbeddingModelSpec{
		Name:       "llama.cpp-embedding",
		Version:    "native-test-v1",
		Dimensions: 4096,
		Metric:     "cosine",
		Normalized: true,
	})
	if err != nil {
		t.Fatalf("first ensure: %v", err)
	}

	secondID, err := EnsureEmbeddingModel(ctx, db, EmbeddingModelSpec{
		Name:       "llama.cpp-embedding",
		Version:    "native-test-v1",
		Dimensions: 4096,
		Metric:     "cosine",
		Normalized: true,
	})
	if err != nil {
		t.Fatalf("second ensure: %v", err)
	}

	if firstID != secondID {
		t.Fatalf("expected same id, got first=%d second=%d", firstID, secondID)
	}
}

func TestEnsureEmbeddingModelRejectsConflictingSpecForExistingNameVersion(t *testing.T) {
	db := openModelDB(t)
	ctx := context.Background()

	_, err := EnsureEmbeddingModel(ctx, db, EmbeddingModelSpec{
		Name:       "llama.cpp-embedding",
		Version:    "native-v2-qwen-mmproj-d2048-s512-t0",
		Dimensions: 2048,
		Metric:     "cosine",
		Normalized: true,
	})
	if err != nil {
		t.Fatalf("seed model: %v", err)
	}

	_, err = EnsureEmbeddingModel(ctx, db, EmbeddingModelSpec{
		Name:       "llama.cpp-embedding",
		Version:    "native-v2-qwen-mmproj-d2048-s512-t0",
		Dimensions: 4096,
		Metric:     "cosine",
		Normalized: true,
	})
	if err == nil {
		t.Fatal("expected conflicting model spec error")
	}
	if !strings.Contains(err.Error(), "does not match requested spec") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestPurgeOtherModelEmbeddingsKeepsOnlyActiveModel(t *testing.T) {
	db := openModelDB(t)
	ctx := context.Background()

	modelA, err := EnsureEmbeddingModel(ctx, db, EmbeddingModelSpec{
		Name:       "llama.cpp-embedding",
		Version:    "native-a",
		Dimensions: 4,
		Metric:     "cosine",
		Normalized: true,
	})
	if err != nil {
		t.Fatalf("ensure model A: %v", err)
	}
	modelB, err := EnsureEmbeddingModel(ctx, db, EmbeddingModelSpec{
		Name:       "llama.cpp-embedding",
		Version:    "native-b",
		Dimensions: 4,
		Metric:     "cosine",
		Normalized: true,
	})
	if err != nil {
		t.Fatalf("ensure model B: %v", err)
	}

	if _, err := db.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES
	(1, 'a', 'one.jpg', 'images/a', 'image/jpeg', 1, 1),
	(2, 'b', 'two.jpg', 'images/b', 'image/jpeg', 1, 1)
`); err != nil {
		t.Fatalf("seed images: %v", err)
	}
	if _, err := db.Exec(`
INSERT INTO image_embeddings(image_id, model_id, dim, vector_blob)
VALUES
	(1, ?, 4, X'00000000'),
	(2, ?, 4, X'00000000')
`, modelA, modelB); err != nil {
		t.Fatalf("seed embeddings: %v", err)
	}

	purged, err := PurgeOtherModelEmbeddings(ctx, db, modelB)
	if err != nil {
		t.Fatalf("purge embeddings: %v", err)
	}
	if purged != 1 {
		t.Fatalf("purged rows: got=%d want=1", purged)
	}

	var countA int
	if err := db.QueryRow(`SELECT COUNT(*) FROM image_embeddings WHERE model_id = ?`, modelA).Scan(&countA); err != nil {
		t.Fatalf("count model A embeddings: %v", err)
	}
	if countA != 0 {
		t.Fatalf("expected model A embeddings to be purged, got %d", countA)
	}

	var countB int
	if err := db.QueryRow(`SELECT COUNT(*) FROM image_embeddings WHERE model_id = ?`, modelB).Scan(&countB); err != nil {
		t.Fatalf("count model B embeddings: %v", err)
	}
	if countB != 1 {
		t.Fatalf("expected model B embedding to remain, got %d", countB)
	}
}
