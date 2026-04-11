package sqlitevector

import (
	"context"
	"database/sql"
	"os"
	"path/filepath"
	"testing"

	sqlite3 "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
)

func TestSQLiteVectorSearchEndToEnd(t *testing.T) {
	if os.Getenv("RUN_SQLITE_VECTOR_INTEGRATION") != "1" {
		t.Skip("set RUN_SQLITE_VECTOR_INTEGRATION=1 to enable")
	}

	extensionPath := os.Getenv("SQLITE_VECTOR_PATH")
	if extensionPath == "" {
		t.Skip("set SQLITE_VECTOR_PATH to sqlite-vector binary path")
	}

	dbConn := openWithSQLiteVector(t, extensionPath)

	if err := db.RunMigrations(context.Background(), dbConn); err != nil {
		t.Fatalf("run migrations: %v", err)
	}

	_, err := dbConn.Exec(`
INSERT INTO embedding_models(id, name, version, dimensions, metric, normalized)
VALUES (1, 'jina-embeddings-v4', 'mlx-8bit', 2, 'cosine', 1)
`)
	if err != nil {
		t.Fatalf("insert model: %v", err)
	}

	_, err = dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES
	(1, 'a', 'one.jpg', 'images/a', 'image/jpeg', 10, 10),
	(2, 'b', 'two.jpg', 'images/b', 'image/jpeg', 10, 10),
	(3, 'c', 'three.jpg', 'images/c', 'image/jpeg', 10, 10)
`)
	if err != nil {
		t.Fatalf("insert images: %v", err)
	}

	idx := NewIndex(dbConn)
	if err := idx.Upsert(context.Background(), 1, 1, []float32{1, 0}); err != nil {
		t.Fatalf("upsert 1: %v", err)
	}
	if err := idx.Upsert(context.Background(), 2, 1, []float32{0.8, 0.2}); err != nil {
		t.Fatalf("upsert 2: %v", err)
	}
	if err := idx.Upsert(context.Background(), 3, 1, []float32{0, 1}); err != nil {
		t.Fatalf("upsert 3: %v", err)
	}

	hits, err := idx.Search(context.Background(), 1, []float32{1, 0}, 3)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(hits) != 3 {
		t.Fatalf("expected 3 hits, got %d", len(hits))
	}
	if hits[0].ImageID != 1 || hits[1].ImageID != 2 {
		t.Fatalf("unexpected hit order: %+v", hits)
	}

	similar, err := idx.SearchByImageID(context.Background(), 1, 1, 2)
	if err != nil {
		t.Fatalf("search by image id: %v", err)
	}
	if len(similar) != 2 {
		t.Fatalf("expected 2 similar hits, got %d", len(similar))
	}
	if similar[0].ImageID != 2 {
		t.Fatalf("expected nearest similar image 2, got %d", similar[0].ImageID)
	}
}

func TestSQLiteVectorSearchKeepsModelIsolation(t *testing.T) {
	if os.Getenv("RUN_SQLITE_VECTOR_INTEGRATION") != "1" {
		t.Skip("set RUN_SQLITE_VECTOR_INTEGRATION=1 to enable")
	}

	extensionPath := os.Getenv("SQLITE_VECTOR_PATH")
	if extensionPath == "" {
		t.Skip("set SQLITE_VECTOR_PATH to sqlite-vector binary path")
	}

	dbConn := openWithSQLiteVector(t, extensionPath)

	if err := db.RunMigrations(context.Background(), dbConn); err != nil {
		t.Fatalf("run migrations: %v", err)
	}

	_, err := dbConn.Exec(`
INSERT INTO embedding_models(id, name, version, dimensions, metric, normalized)
VALUES
	(1, 'jina-embeddings-v4', 'mlx-8bit', 2, 'cosine', 1),
	(2, 'deterministic-sha256', 'v1', 2, 'cosine', 1)
`)
	if err != nil {
		t.Fatalf("insert models: %v", err)
	}

	_, err = dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES
	(1, 'a', 'one.jpg', 'images/a', 'image/jpeg', 10, 10),
	(2, 'b', 'two.jpg', 'images/b', 'image/jpeg', 10, 10)
`)
	if err != nil {
		t.Fatalf("insert images: %v", err)
	}

	idx := NewIndex(dbConn)

	// Model 1 vectors are far from query [1, 0].
	if err := idx.Upsert(context.Background(), 1, 1, []float32{0, 1}); err != nil {
		t.Fatalf("upsert model1 image1: %v", err)
	}
	if err := idx.Upsert(context.Background(), 2, 1, []float32{0, -1}); err != nil {
		t.Fatalf("upsert model1 image2: %v", err)
	}

	// Model 2 vectors are near query [1, 0].
	if err := idx.Upsert(context.Background(), 1, 2, []float32{1, 0}); err != nil {
		t.Fatalf("upsert model2 image1: %v", err)
	}
	if err := idx.Upsert(context.Background(), 2, 2, []float32{0.99, 0.01}); err != nil {
		t.Fatalf("upsert model2 image2: %v", err)
	}

	hits, err := idx.Search(context.Background(), 1, []float32{1, 0}, 2)
	if err != nil {
		t.Fatalf("search model1: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("expected 2 hits for model1, got %d", len(hits))
	}
	for _, hit := range hits {
		if hit.ModelID != 1 {
			t.Fatalf("expected hit model_id=1, got %d", hit.ModelID)
		}
	}
}

func openWithSQLiteVector(t *testing.T, extensionPath string) *sql.DB {
	t.Helper()
	absPath, err := filepath.Abs(extensionPath)
	if err != nil {
		t.Fatalf("resolve extension path: %v", err)
	}

	dbConn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	dbConn.SetMaxOpenConns(1)
	dbConn.SetMaxIdleConns(1)
	t.Cleanup(func() { _ = dbConn.Close() })

	loadTarget := absPath
	ext := filepath.Ext(loadTarget)
	if ext == ".dylib" || ext == ".so" || ext == ".dll" {
		loadTarget = loadTarget[:len(loadTarget)-len(ext)]
	}

	conn, err := dbConn.Conn(context.Background())
	if err != nil {
		t.Fatalf("acquire sqlite conn: %v", err)
	}

	if err := conn.Raw(func(driverConn any) error {
		sqliteConn, ok := driverConn.(*sqlite3.SQLiteConn)
		if !ok {
			return sql.ErrConnDone
		}
		return sqliteConn.LoadExtension(loadTarget, "sqlite3_vector_init")
	}); err != nil {
		_ = conn.Close()
		t.Fatalf("load sqlite-vector extension from %q: %v", loadTarget, err)
	}
	if err := conn.Close(); err != nil {
		t.Fatalf("close sqlite conn: %v", err)
	}

	if err := ValidateAvailable(context.Background(), dbConn); err != nil {
		t.Fatalf("validate sqlite-vector: %v", err)
	}

	return dbConn
}
