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

func setupReannotateDB(t *testing.T) *sql.DB {
	t.Helper()
	conn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	conn.SetMaxOpenConns(1)
	t.Cleanup(func() { _ = conn.Close() })
	if err := db.RunMigrations(context.Background(), conn); err != nil {
		t.Fatalf("run migrations: %v", err)
	}
	if _, err := conn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height, title, summary, description, tags_json)
VALUES
	(1, 'a', 'one.jpg', 'images/a', 'image/jpeg', 10, 10, 'T1', 'S1', 'D1', '["x"]'),
	(2, 'b', 'two.jpg', 'images/b', 'image/jpeg', 10, 10, '', '', '', '[]'),
	(3, 'c', 'frame.jpg', 'images/c', 'image/jpeg', 10, 10, 'F', 'F', 'frame', '["f"]');
INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, width, height, duration_ms, frame_count, title, summary, description, tags_json)
VALUES
	(7, 'v', 'clip.mp4', 'videos/v', 'video/mp4', 10, 10, 1000, 1, 'VT', 'VS', 'VD', '["v"]');
INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms) VALUES (7, 3, 0, 0);
INSERT INTO index_jobs(id, kind, image_id, video_id, model_id, state, attempts, max_attempts, last_error)
VALUES
	(11, 'annotate_image', 1, NULL, 1, 'done', 1, 3, NULL),
	(12, 'annotate_image', 2, NULL, 1, 'failed', 3, 3, 'oom'),
	(13, 'annotate_video', NULL, 7, 1, 'leased', 1, 3, NULL),
	(14, 'annotate_image', 1, NULL, 2, 'done', 1, 3, NULL)
`); err != nil {
		t.Fatalf("seed: %v", err)
	}
	return conn
}

func TestReannotateAllQueuesStandaloneImagesAndVideosKeepingText(t *testing.T) {
	conn := setupReannotateDB(t)
	h := NewReannotateAllHandler(&ReannotateAllHandler{DB: conn, ModelID: 1})

	req := httptest.NewRequest(http.MethodPost, "/api/jobs/reannotate-all", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}
	var resp ReannotateAllResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.QueuedImages != 2 || resp.QueuedVideos != 0 || resp.SkippedLeased != 1 {
		t.Fatalf("unexpected counts: %+v", resp)
	}

	var pendingImages, flaggedImages, flaggedFrames, flaggedVideos int
	if err := conn.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE kind = 'annotate_image' AND model_id = 1 AND state = 'pending' AND attempts = 0`).Scan(&pendingImages); err != nil {
		t.Fatal(err)
	}
	if pendingImages != 2 {
		t.Fatalf("expected both standalone image jobs pending, got %d", pendingImages)
	}
	if err := conn.QueryRow(`SELECT COUNT(*) FROM images WHERE reannotate_requested = 1 AND id IN (1, 2)`).Scan(&flaggedImages); err != nil {
		t.Fatal(err)
	}
	if err := conn.QueryRow(`SELECT COUNT(*) FROM images WHERE reannotate_requested = 1 AND id = 3`).Scan(&flaggedFrames); err != nil {
		t.Fatal(err)
	}
	if flaggedImages != 2 || flaggedFrames != 0 {
		t.Fatalf("expected standalone images flagged and frames untouched: images=%d frames=%d", flaggedImages, flaggedFrames)
	}
	if err := conn.QueryRow(`SELECT COUNT(*) FROM videos WHERE reannotate_requested = 1`).Scan(&flaggedVideos); err != nil {
		t.Fatal(err)
	}
	if flaggedVideos != 1 {
		t.Fatalf("expected video flagged even while its job is leased, got %d", flaggedVideos)
	}

	var title, description string
	if err := conn.QueryRow(`SELECT title, description FROM images WHERE id = 1`).Scan(&title, &description); err != nil {
		t.Fatal(err)
	}
	if title != "T1" || description != "D1" {
		t.Fatalf("existing text must be kept until replaced, got title=%q description=%q", title, description)
	}
	var otherModelState string
	if err := conn.QueryRow(`SELECT state FROM index_jobs WHERE id = 14`).Scan(&otherModelState); err != nil {
		t.Fatal(err)
	}
	if otherModelState != "done" {
		t.Fatalf("other model's job must be untouched, got %q", otherModelState)
	}
	var leasedState string
	if err := conn.QueryRow(`SELECT state FROM index_jobs WHERE id = 13`).Scan(&leasedState); err != nil {
		t.Fatal(err)
	}
	if leasedState != "leased" {
		t.Fatalf("leased job must not be reset, got %q", leasedState)
	}
}

func TestReannotateAllHonoursMediaFilterAndIsIdempotent(t *testing.T) {
	conn := setupReannotateDB(t)
	if _, err := conn.Exec(`UPDATE index_jobs SET state = 'done' WHERE id = 13`); err != nil {
		t.Fatal(err)
	}
	h := NewReannotateAllHandler(&ReannotateAllHandler{DB: conn, ModelID: 1})

	call := func(query string) ReannotateAllResponse {
		t.Helper()
		req := httptest.NewRequest(http.MethodPost, "/api/jobs/reannotate-all"+query, nil)
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, req)
		if rr.Code != http.StatusOK {
			t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
		}
		var resp ReannotateAllResponse
		if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
			t.Fatal(err)
		}
		return resp
	}

	videos := call("?media=videos")
	if videos.QueuedImages != 0 || videos.QueuedVideos != 1 {
		t.Fatalf("videos filter: %+v", videos)
	}
	var imagesFlagged int
	if err := conn.QueryRow(`SELECT COUNT(*) FROM images WHERE reannotate_requested = 1`).Scan(&imagesFlagged); err != nil {
		t.Fatal(err)
	}
	if imagesFlagged != 0 {
		t.Fatalf("videos filter must not touch images, got %d flagged", imagesFlagged)
	}

	all := call("")
	again := call("?media=all")
	if all.QueuedImages != 2 || all.QueuedVideos != 1 || again.QueuedImages != 2 || again.QueuedVideos != 1 {
		t.Fatalf("expected idempotent counts, got first=%+v second=%+v", all, again)
	}
	var pending int
	if err := conn.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE model_id = 1 AND state = 'pending'`).Scan(&pending); err != nil {
		t.Fatal(err)
	}
	if pending != 3 {
		t.Fatalf("expected 3 pending annotation jobs, got %d", pending)
	}
}

func TestReannotateAllRejectsBadRequests(t *testing.T) {
	conn := setupReannotateDB(t)
	h := NewReannotateAllHandler(&ReannotateAllHandler{DB: conn, ModelID: 1})

	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/api/jobs/reannotate-all", nil))
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", rr.Code)
	}
	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/api/jobs/reannotate-all?media=audio", nil))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for bad media filter, got %d", rr.Code)
	}
}
