package sqlitevector

import (
	"context"
	"database/sql"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
)

func TestEmbeddingSnapshotReturnsModelAndTableStats(t *testing.T) {
	dbConn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = dbConn.Close() })

	if err := db.RunMigrations(context.Background(), dbConn); err != nil {
		t.Fatalf("run migrations: %v", err)
	}

	if _, err := dbConn.Exec(`
INSERT INTO embedding_models(id, name, version, dimensions, metric, normalized)
VALUES
	(1, 'model', 'a', 4, 'cosine', 1),
	(2, 'model', 'b', 4, 'cosine', 1)
`); err != nil {
		t.Fatalf("seed models: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES
	(1, 'a', 'one.jpg', 'images/a', 'image/jpeg', 1, 1),
	(2, 'b', 'two.jpg', 'images/b', 'image/jpeg', 1, 1),
	(3, 'c', 'three.jpg', 'images/c', 'image/jpeg', 1, 1)
`); err != nil {
		t.Fatalf("seed images: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO image_embeddings(image_id, model_id, dim, vector_blob, updated_at)
VALUES
	(1, 1, 4, X'00000000', '2026-01-01 00:00:00'),
	(2, 1, 4, X'00000000', '2026-01-01 00:00:05'),
	(3, 2, 4, X'00000000', '2026-01-01 00:00:03')
`); err != nil {
		t.Fatalf("seed embeddings: %v", err)
	}

	index := NewIndex(dbConn)
	snapshot, err := index.embeddingSnapshot(context.Background(), 1)
	if err != nil {
		t.Fatalf("embedding snapshot: %v", err)
	}

	if snapshot.modelCount != 2 {
		t.Fatalf("model count: got=%d want=2", snapshot.modelCount)
	}
	if snapshot.totalCount != 3 {
		t.Fatalf("total count: got=%d want=3", snapshot.totalCount)
	}
	if snapshot.latestUpdatedAt != "2026-01-01 00:00:05" {
		t.Fatalf("latest updated_at: got=%q want=%q", snapshot.latestUpdatedAt, "2026-01-01 00:00:05")
	}
}

func TestNeedsQuantizationRefreshWhenSnapshotChanges(t *testing.T) {
	index := NewIndex(nil)
	modelID := int64(7)
	snapshot := quantizationSnapshot{
		totalCount:      12,
		latestUpdatedAt: "2026-01-01 00:00:05",
	}

	if !index.needsQuantizationRefresh(modelID, snapshot) {
		t.Fatalf("expected initial snapshot to require quantization")
	}

	index.quantized[modelID] = snapshot
	if index.needsQuantizationRefresh(modelID, snapshot) {
		t.Fatalf("expected identical snapshot to skip re-quantization")
	}

	changedCount := quantizationSnapshot{
		totalCount:      snapshot.totalCount + 1,
		latestUpdatedAt: snapshot.latestUpdatedAt,
	}
	if !index.needsQuantizationRefresh(modelID, changedCount) {
		t.Fatalf("expected count change to require re-quantization")
	}

	changedTimestamp := quantizationSnapshot{
		totalCount:      snapshot.totalCount,
		latestUpdatedAt: "2026-01-01 00:00:06",
	}
	if !index.needsQuantizationRefresh(modelID, changedTimestamp) {
		t.Fatalf("expected updated_at change to require re-quantization")
	}
}

func TestQuantizedKUsesRequestedLimitAndClampsToModelCount(t *testing.T) {
	if got := quantizedK(25, 20000); got != 25 {
		t.Fatalf("expected k=25 for limit=25 model_count=20000, got %d", got)
	}
	if got := quantizedK(200, 120); got != 120 {
		t.Fatalf("expected k clamped to model count 120, got %d", got)
	}
	if got := quantizedK(0, 15); got != 15 {
		t.Fatalf("expected default k clamped to model count 15, got %d", got)
	}
	if got := quantizedK(0, 0); got != 0 {
		t.Fatalf("expected k=0 with no embeddings, got %d", got)
	}
}

func TestCountEmbeddingsScopesToModelID(t *testing.T) {
	dbConn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = dbConn.Close() })

	if err := db.RunMigrations(context.Background(), dbConn); err != nil {
		t.Fatalf("run migrations: %v", err)
	}

	if _, err := dbConn.Exec(`
INSERT INTO embedding_models(id, name, version, dimensions, metric, normalized)
VALUES
	(1, 'model', 'a', 4, 'cosine', 1),
	(2, 'model', 'b', 4, 'cosine', 1)
`); err != nil {
		t.Fatalf("seed models: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES
	(1, 'a', 'one.jpg', 'images/a', 'image/jpeg', 1, 1),
	(2, 'b', 'two.jpg', 'images/b', 'image/jpeg', 1, 1),
	(3, 'c', 'three.jpg', 'images/c', 'image/jpeg', 1, 1)
`); err != nil {
		t.Fatalf("seed images: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO image_embeddings(image_id, model_id, dim, vector_blob)
VALUES
	(1, 1, 4, X'00000000'),
	(2, 1, 4, X'00000000'),
	(3, 2, 4, X'00000000')
`); err != nil {
		t.Fatalf("seed embeddings: %v", err)
	}

	index := NewIndex(dbConn)
	total, err := index.countEmbeddings(context.Background(), 1)
	if err != nil {
		t.Fatalf("count embeddings: %v", err)
	}
	if total != 2 {
		t.Fatalf("count embeddings: got=%d want=2", total)
	}
}
