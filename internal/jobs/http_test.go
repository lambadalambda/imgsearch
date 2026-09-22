package jobs

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

func setupJobsDB(t *testing.T) *sql.DB {
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
	(4, 'd', 'four.jpg', 'images/d', 'image/jpeg', 10, 10)
`)
	if err != nil {
		t.Fatalf("seed images: %v", err)
	}

	_, err = dbConn.Exec(`
INSERT INTO index_jobs(id, kind, image_id, model_id, state, attempts, max_attempts, last_error)
VALUES
	(11, 'embed_image', 1, 1, 'failed', 3, 3, 'oom'),
	(12, 'embed_image', 2, 1, 'failed', 2, 3, 'timeout'),
	(13, 'annotate_image', 3, 1, 'done', 1, 3, NULL),
	(15, 'annotate_image', 2, 1, 'failed', 2, 3, 'annotator oom'),
	(14, 'embed_image', 1, 2, 'failed', 3, 3, 'other model')
`)
	if err != nil {
		t.Fatalf("seed jobs: %v", err)
	}

	return dbConn
}

func TestRetryFailedHandlerResetsFailedJobsForModel(t *testing.T) {
	dbConn := setupJobsDB(t)
	h := NewRetryFailedHandler(&RetryFailedHandler{DB: dbConn, ModelID: 1})

	req := httptest.NewRequest(http.MethodPost, "/api/jobs/retry-failed", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var resp RetryFailedResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Retried != 3 {
		t.Fatalf("retried: got=%d want=3", resp.Retried)
	}
	if resp.Enqueued != 5 {
		t.Fatalf("enqueued_missing: got=%d want=5", resp.Enqueued)
	}

	rows, err := dbConn.Query(`
SELECT state, attempts, COALESCE(last_error, ''), COALESCE(run_after, '')
FROM index_jobs
WHERE id IN (11, 12, 15)
ORDER BY id ASC
`)
	if err != nil {
		t.Fatalf("query retried jobs: %v", err)
	}
	defer func() { _ = rows.Close() }()

	for rows.Next() {
		var state string
		var attempts int
		var lastError string
		var runAfter string
		if err := rows.Scan(&state, &attempts, &lastError, &runAfter); err != nil {
			t.Fatalf("scan retried row: %v", err)
		}
		if state != "pending" {
			t.Fatalf("expected pending state, got %s", state)
		}
		if attempts != 0 {
			t.Fatalf("expected attempts reset to 0, got %d", attempts)
		}
		if lastError != "" {
			t.Fatalf("expected empty last_error, got %q", lastError)
		}
		if runAfter != "" {
			t.Fatalf("expected empty run_after, got %q", runAfter)
		}
	}

	var otherModelState string
	if err := dbConn.QueryRow(`SELECT state FROM index_jobs WHERE id = 14`).Scan(&otherModelState); err != nil {
		t.Fatalf("query other model state: %v", err)
	}
	if otherModelState != "failed" {
		t.Fatalf("expected model 2 failed to remain unchanged, got %s", otherModelState)
	}

	var missingState string
	if err := dbConn.QueryRow(`SELECT state FROM index_jobs WHERE kind='embed_image' AND model_id = 1 AND image_id = 4`).Scan(&missingState); err != nil {
		t.Fatalf("query missing job after retry: %v", err)
	}
	if missingState != "pending" {
		t.Fatalf("expected missing image to be queued as pending, got %s", missingState)
	}

	var annotationRepairState string
	if err := dbConn.QueryRow(`SELECT state FROM index_jobs WHERE id = 13`).Scan(&annotationRepairState); err != nil {
		t.Fatalf("query annotation repair state: %v", err)
	}
	if annotationRepairState != "pending" {
		t.Fatalf("expected done image missing annotations to be requeued, got %s", annotationRepairState)
	}

	var missingAnnotateState string
	if err := dbConn.QueryRow(`SELECT state FROM index_jobs WHERE kind='annotate_image' AND model_id = 1 AND image_id = 4`).Scan(&missingAnnotateState); err != nil {
		t.Fatalf("query missing annotate job after retry: %v", err)
	}
	if missingAnnotateState != "pending" {
		t.Fatalf("expected missing annotate job to be queued as pending, got %s", missingAnnotateState)
	}
}

func TestRetryFailedHandlerEnqueuesMissingVideoTranscriptJobs(t *testing.T) {
	dbConn := setupJobsDB(t)
	if _, err := dbConn.Exec(`
INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count)
VALUES (1, 'video-a', 'clip.mp4', 'videos/video-a', 'video/mp4', 1000, 640, 360, 0)
`); err != nil {
		t.Fatalf("seed video: %v", err)
	}

	h := NewRetryFailedHandler(&RetryFailedHandler{DB: dbConn, ModelID: 1})
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/retry-failed", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var state string
	if err := dbConn.QueryRow(`
SELECT state
FROM index_jobs
WHERE kind = 'transcribe_video'
  AND video_id = 1
  AND image_id IS NULL
  AND model_id = 1
`).Scan(&state); err != nil {
		t.Fatalf("query transcript job: %v", err)
	}
	if state != "pending" {
		t.Fatalf("expected missing transcript job pending, got %s", state)
	}
}

func TestRetryFailedHandlerRejectsInvalidMethod(t *testing.T) {
	dbConn := setupJobsDB(t)
	h := NewRetryFailedHandler(&RetryFailedHandler{DB: dbConn, ModelID: 1})

	req := httptest.NewRequest(http.MethodGet, "/api/jobs/retry-failed", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status: got=%d want=%d", rr.Code, http.StatusMethodNotAllowed)
	}
}

func TestFilteredRetryOnlyExistingFailedKind(t *testing.T) {
	for _, tc := range []struct {
		query           string
		blocked         bool
		status, retried int
	}{
		{"?kind=embed_image", false, 200, 2},
		{"?kind=annotate_image", false, 200, 1},
		{"?kind=transcribe_video", true, 409, 0},
		{"?kind=typo", false, 400, 0},
		{"?kind=", false, 400, 0},
		{"?kind=annotate_image%ZZ", false, 400, 0},
		{"?kind=annotate_image;", false, 400, 0},
	} {
		t.Run(tc.query, func(t *testing.T) {
			conn := setupJobsDB(t)
			h := NewRetryFailedHandler(&RetryFailedHandler{DB: conn, ModelID: 1, BlockedKinds: map[string]bool{"transcribe_video": tc.blocked}})
			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, httptest.NewRequest("POST", "/api/jobs/retry-failed"+tc.query, nil))
			if rr.Code != tc.status {
				t.Fatalf("status %d: %s", rr.Code, rr.Body.String())
			}
			var count int
			if err := conn.QueryRow(`SELECT COUNT(*) FROM index_jobs`).Scan(&count); err != nil {
				t.Fatal(err)
			}
			if count != 5 {
				t.Fatalf("filtered retry inserted jobs: %d", count)
			}
			if err := conn.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE state='pending' AND model_id=1`).Scan(&count); err != nil {
				t.Fatal(err)
			}
			if count != tc.retried {
				t.Fatalf("retried=%d want=%d", count, tc.retried)
			}
			if err := conn.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE id IN (13,14) AND state IN ('done','failed')`).Scan(&count); err != nil {
				t.Fatal(err)
			}
			if count != 2 {
				t.Fatal("changed done or other-model job")
			}
			if tc.status == 200 {
				var got RetryFailedResponse
				_ = json.Unmarshal(rr.Body.Bytes(), &got)
				if got.Enqueued != 0 || got.Retried != int64(tc.retried) {
					t.Fatalf("response %+v", got)
				}
			}
		})
	}
}
