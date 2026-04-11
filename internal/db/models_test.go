package db

import (
	"context"
	"database/sql"
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
		Name:       "jina-embeddings-v4",
		Version:    "mlx-8bit",
		Dimensions: 2048,
		Metric:     "cosine",
		Normalized: true,
	})
	if err != nil {
		t.Fatalf("first ensure: %v", err)
	}

	secondID, err := EnsureEmbeddingModel(ctx, db, EmbeddingModelSpec{
		Name:       "jina-embeddings-v4",
		Version:    "mlx-8bit",
		Dimensions: 2048,
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
