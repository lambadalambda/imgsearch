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

func TestRequeueDoneJobsMissingAnnotations(t *testing.T) {
	dbConn := openIndexJobsDB(t)

	if _, err := dbConn.Exec(`
UPDATE images
SET description = 'already annotated', tags_json = '["tagged"]'
WHERE id = 1
`); err != nil {
		t.Fatalf("seed annotations: %v", err)
	}

	if _, err := dbConn.Exec(`
INSERT INTO index_jobs(kind, image_id, model_id, state)
VALUES
	('embed_image', 2, 1, 'done'),
	('embed_image', 3, 1, 'done')
`); err != nil {
		t.Fatalf("seed extra jobs: %v", err)
	}

	rows, err := RequeueDoneJobsMissingAnnotations(context.Background(), dbConn, 1)
	if err != nil {
		t.Fatalf("requeue missing annotations: %v", err)
	}
	if rows != 2 {
		t.Fatalf("requeued rows: got=%d want=2", rows)
	}

	var state1 string
	if err := dbConn.QueryRow(`SELECT state FROM index_jobs WHERE image_id = 1 AND model_id = 1`).Scan(&state1); err != nil {
		t.Fatalf("load annotated job state: %v", err)
	}
	if state1 != "done" {
		t.Fatalf("expected annotated image job to stay done, got %s", state1)
	}

	var pendingCount int
	if err := dbConn.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE model_id = 1 AND state = 'pending'`).Scan(&pendingCount); err != nil {
		t.Fatalf("count pending jobs: %v", err)
	}
	if pendingCount != 2 {
		t.Fatalf("expected 2 pending jobs after requeue, got %d", pendingCount)
	}
}

func TestPurgeOtherModelIndexJobsKeepsOnlyActiveModel(t *testing.T) {
	dbConn := openIndexJobsDB(t)

	if _, err := dbConn.Exec(`
INSERT INTO index_jobs(kind, image_id, model_id, state)
VALUES
	('embed_image', 2, 2, 'pending'),
	('embed_image', 3, 1, 'pending')
`); err != nil {
		t.Fatalf("seed extra jobs: %v", err)
	}

	purged, err := PurgeOtherModelIndexJobs(context.Background(), dbConn, 1)
	if err != nil {
		t.Fatalf("purge index jobs: %v", err)
	}
	if purged != 1 {
		t.Fatalf("purged rows: got=%d want=1", purged)
	}

	var activeCount int
	if err := dbConn.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE model_id = 1`).Scan(&activeCount); err != nil {
		t.Fatalf("count active jobs: %v", err)
	}
	if activeCount != 2 {
		t.Fatalf("active job count: got=%d want=2", activeCount)
	}

	var otherCount int
	if err := dbConn.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE model_id = 2`).Scan(&otherCount); err != nil {
		t.Fatalf("count other-model jobs: %v", err)
	}
	if otherCount != 0 {
		t.Fatalf("expected other-model jobs to be purged, got %d", otherCount)
	}
}
