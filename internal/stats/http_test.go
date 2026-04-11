package stats

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
)

func setupStatsDB(t *testing.T) *sql.DB {
	t.Helper()

	dbConn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = dbConn.Close() })

	if err := db.RunMigrations(context.Background(), dbConn); err != nil {
		t.Fatalf("run migrations: %v", err)
	}

	_, err = dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES
	(1, 'a', 'one.jpg', 'images/a', 'image/jpeg', 10, 10),
	(2, 'b', 'two.jpg', 'images/b', 'image/jpeg', 10, 10),
	(3, 'c', 'three.jpg', 'images/c', 'image/jpeg', 10, 10),
	(4, 'd', 'four.jpg', 'images/d', 'image/jpeg', 10, 10),
	(5, 'e', 'five.jpg', 'images/e', 'image/jpeg', 10, 10)
`)
	if err != nil {
		t.Fatalf("seed images: %v", err)
	}

	_, err = dbConn.Exec(`
INSERT INTO index_jobs(id, kind, image_id, model_id, state, attempts, max_attempts, last_error)
VALUES
	(11, 'embed_image', 1, 1, 'done', 1, 3, NULL),
	(12, 'embed_image', 2, 1, 'pending', 1, 3, NULL),
	(13, 'embed_image', 3, 1, 'leased', 2, 3, NULL),
	(14, 'embed_image', 4, 1, 'failed', 3, 3, 'oom crash'),
	(15, 'embed_image', 1, 2, 'failed', 3, 3, 'other model')
`)
	if err != nil {
		t.Fatalf("seed jobs: %v", err)
	}

	return dbConn
}

func TestStatsHandlerReturnsQueueCountsAndFailures(t *testing.T) {
	dbConn := setupStatsDB(t)
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})

	req := httptest.NewRequest(http.MethodGet, "/api/stats", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var resp Response
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}

	if resp.ImagesTotal != 5 {
		t.Fatalf("images_total: got=%d want=5", resp.ImagesTotal)
	}
	if resp.Queue.Tracked != 4 {
		t.Fatalf("queue.tracked: got=%d want=4", resp.Queue.Tracked)
	}
	if resp.Queue.Missing != 1 {
		t.Fatalf("queue.missing: got=%d want=1", resp.Queue.Missing)
	}
	if resp.Queue.Total != 5 {
		t.Fatalf("queue.total: got=%d want=5", resp.Queue.Total)
	}
	if resp.Queue.Done != 1 || resp.Queue.Pending != 1 || resp.Queue.Leased != 1 || resp.Queue.Failed != 1 {
		t.Fatalf("unexpected queue counts: %+v", resp.Queue)
	}
	if len(resp.RecentFailures) != 1 {
		t.Fatalf("recent_failures length: got=%d want=1", len(resp.RecentFailures))
	}
	if resp.RecentFailures[0].JobID != 14 {
		t.Fatalf("failure job id: got=%d want=14", resp.RecentFailures[0].JobID)
	}
	if resp.RecentFailures[0].OriginalName != "four.jpg" {
		t.Fatalf("failure original name: got=%q", resp.RecentFailures[0].OriginalName)
	}
}

func TestStatsHandlerRejectsInvalidMethod(t *testing.T) {
	dbConn := setupStatsDB(t)
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})

	req := httptest.NewRequest(http.MethodPost, "/api/stats", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status: got=%d want=%d", rr.Code, http.StatusMethodNotAllowed)
	}
}
