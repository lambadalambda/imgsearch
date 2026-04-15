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
INSERT INTO index_jobs(id, kind, image_id, model_id, state, leased_until, attempts, max_attempts, last_error)
VALUES
	(11, 'embed_image', 1, 1, 'done', NULL, 1, 3, NULL),
	(12, 'embed_image', 2, 1, 'pending', NULL, 1, 3, NULL),
	(13, 'embed_image', 3, 1, 'leased', datetime('now', '-5 minutes'), 2, 3, NULL),
	(14, 'embed_image', 4, 1, 'failed', NULL, 3, 3, 'oom crash'),
	(16, 'annotate_image', 2, 1, 'pending', NULL, 0, 3, NULL),
	(17, 'annotate_image', 3, 1, 'failed', NULL, 2, 3, 'annotator timeout'),
	(15, 'embed_image', 1, 2, 'failed', NULL, 3, 3, 'other model')
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
	if resp.Queue.Runnable != 2 {
		t.Fatalf("queue.runnable: got=%d want=2", resp.Queue.Runnable)
	}
	if resp.Queue.OldestRunnableAgeSeconds < 240 {
		t.Fatalf("queue.oldest_runnable_age_seconds: got=%d want>=240", resp.Queue.OldestRunnableAgeSeconds)
	}
	if resp.Queue.Done != 1 || resp.Queue.Pending != 1 || resp.Queue.Leased != 1 || resp.Queue.Failed != 1 {
		t.Fatalf("unexpected queue counts: %+v", resp.Queue)
	}
	if len(resp.JobKinds) != 2 {
		t.Fatalf("job_kinds length: got=%d want=2", len(resp.JobKinds))
	}
	if got := resp.JobKinds["embed_image"]; got.Tracked != 4 || got.Runnable != 2 || got.Failed != 1 || got.OldestRunnableAgeSeconds < 240 {
		t.Fatalf("unexpected embed_image stats: %+v", got)
	}
	if got := resp.JobKinds["annotate_image"]; got.Tracked != 2 || got.Runnable != 1 || got.Failed != 1 {
		t.Fatalf("unexpected annotate_image stats: %+v", got)
	}
	if len(resp.RecentFailures) != 2 {
		t.Fatalf("recent_failures length: got=%d want=2", len(resp.RecentFailures))
	}
	failuresByID := map[int64]FailureItem{}
	for _, item := range resp.RecentFailures {
		failuresByID[item.JobID] = item
	}
	if got, ok := failuresByID[14]; !ok {
		t.Fatalf("expected embed_image failure for job 14, got=%v", resp.RecentFailures)
	} else {
		if got.Kind != "embed_image" {
			t.Fatalf("failure job kind: got=%q want=embed_image", got.Kind)
		}
		if got.OriginalName != "four.jpg" {
			t.Fatalf("failure original name: got=%q", got.OriginalName)
		}
	}
	if got, ok := failuresByID[17]; !ok {
		t.Fatalf("expected annotate_image failure for job 17, got=%v", resp.RecentFailures)
	} else if got.Kind != "annotate_image" {
		t.Fatalf("annotate failure job kind: got=%q want=annotate_image", got.Kind)
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
