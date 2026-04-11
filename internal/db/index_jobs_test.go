package db

import (
	"context"
	"database/sql"
	"testing"

	_ "github.com/mattn/go-sqlite3"
)

func openIndexJobsDB(t *testing.T) *sql.DB {
	t.Helper()

	dbConn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = dbConn.Close() })

	if err := RunMigrations(context.Background(), dbConn); err != nil {
		t.Fatalf("run migrations: %v", err)
	}

	_, err = dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES
	(1, 'a', 'one.jpg', 'images/a', 'image/jpeg', 10, 10),
	(2, 'b', 'two.jpg', 'images/b', 'image/jpeg', 10, 10),
	(3, 'c', 'three.jpg', 'images/c', 'image/jpeg', 10, 10)
`)
	if err != nil {
		t.Fatalf("seed images: %v", err)
	}

	_, err = dbConn.Exec(`
INSERT INTO index_jobs(kind, image_id, model_id, state)
VALUES ('embed_image', 1, 1, 'done')
`)
	if err != nil {
		t.Fatalf("seed jobs: %v", err)
	}

	return dbConn
}

func TestEnsureIndexJobsForModelEnqueuesMissingRows(t *testing.T) {
	dbConn := openIndexJobsDB(t)

	inserted, err := EnsureIndexJobsForModel(context.Background(), dbConn, 1)
	if err != nil {
		t.Fatalf("ensure jobs: %v", err)
	}
	if inserted != 2 {
		t.Fatalf("inserted: got=%d want=2", inserted)
	}

	var count int64
	if err := dbConn.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE model_id = 1`).Scan(&count); err != nil {
		t.Fatalf("count jobs: %v", err)
	}
	if count != 3 {
		t.Fatalf("job count: got=%d want=3", count)
	}

	insertedAgain, err := EnsureIndexJobsForModel(context.Background(), dbConn, 1)
	if err != nil {
		t.Fatalf("ensure jobs second run: %v", err)
	}
	if insertedAgain != 0 {
		t.Fatalf("inserted second run: got=%d want=0", insertedAgain)
	}
}
