package mediaops

import (
	"context"
	"database/sql"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
)

func setupMediaOpsDB(t *testing.T) *sql.DB {
	t.Helper()
	dbConn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = dbConn.Close() })
	if err := db.RunMigrations(context.Background(), dbConn); err != nil {
		t.Fatalf("run migrations: %v", err)
	}
	return dbConn
}

func TestRequestReannotationJobResetsImageJob(t *testing.T) {
	dbConn := setupMediaOpsDB(t)
	if _, err := dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES (1, 'img1', 'one.jpg', 'images/one', 'image/jpeg', 10, 10)
`); err != nil {
		t.Fatalf("seed image: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO index_jobs(kind, image_id, model_id, state, attempts, last_error)
VALUES ('annotate_image', 1, 7, 'done', 3, 'old error')
`); err != nil {
		t.Fatalf("seed image job: %v", err)
	}

	tx, err := dbConn.BeginTx(context.Background(), nil)
	if err != nil {
		t.Fatalf("begin tx: %v", err)
	}
	if err := RequestReannotationJob(context.Background(), tx, ReannotationTarget{Kind: "annotate_image", ImageID: 1, ModelID: 7}); err != nil {
		_ = tx.Rollback()
		t.Fatalf("request reannotation: %v", err)
	}
	if err := tx.Commit(); err != nil {
		t.Fatalf("commit tx: %v", err)
	}

	var state string
	var attempts int
	var lastError sql.NullString
	if err := dbConn.QueryRow(`
SELECT state, attempts, last_error
FROM index_jobs
WHERE kind = 'annotate_image' AND image_id = 1 AND model_id = 7
`).Scan(&state, &attempts, &lastError); err != nil {
		t.Fatalf("load job: %v", err)
	}
	if state != "pending" || attempts != 0 || lastError.Valid {
		t.Fatalf("unexpected reset job: state=%s attempts=%d last_error=%v", state, attempts, lastError)
	}
}

func TestRequestReannotationJobCreatesVideoJob(t *testing.T) {
	dbConn := setupMediaOpsDB(t)
	if _, err := dbConn.Exec(`
INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count)
VALUES (2, 'vid2', 'two.mp4', 'videos/two', 'video/mp4', 1000, 10, 10, 1)
`); err != nil {
		t.Fatalf("seed video: %v", err)
	}

	tx, err := dbConn.BeginTx(context.Background(), nil)
	if err != nil {
		t.Fatalf("begin tx: %v", err)
	}
	if err := RequestReannotationJob(context.Background(), tx, ReannotationTarget{Kind: "annotate_video", VideoID: 2, ModelID: 7}); err != nil {
		_ = tx.Rollback()
		t.Fatalf("request reannotation: %v", err)
	}
	if err := tx.Commit(); err != nil {
		t.Fatalf("commit tx: %v", err)
	}

	var state string
	if err := dbConn.QueryRow(`
SELECT state
FROM index_jobs
WHERE kind = 'annotate_video' AND video_id = 2 AND model_id = 7
`).Scan(&state); err != nil {
		t.Fatalf("load job: %v", err)
	}
	if state != "pending" {
		t.Fatalf("unexpected job state: %s", state)
	}
}
