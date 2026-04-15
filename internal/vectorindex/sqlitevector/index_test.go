package sqlitevector

import (
	"context"
	"database/sql"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
)

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
