package videos

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

func setupVideosDB(t *testing.T) *sql.DB {
	t.Helper()

	dbConn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = dbConn.Close() })

	if err := db.RunMigrations(context.Background(), dbConn); err != nil {
		t.Fatalf("run migrations: %v", err)
	}

	if _, err := dbConn.Exec(`
INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count)
VALUES
	(1, 'v1', 'first.mp4', 'videos/v1', 'video/mp4', 12000, 1920, 1080, 2),
	(2, 'v2', 'second.mov', 'videos/v2', 'video/quicktime', 8000, 1280, 720, 1)
`); err != nil {
		t.Fatalf("seed videos: %v", err)
	}
	if _, err := dbConn.Exec(`UPDATE videos SET transcript_text = 'tis better to remain silent' WHERE id = 1`); err != nil {
		t.Fatalf("seed transcript text: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES
	(10, 'f1', 'frame1.jpg', 'images/f1', 'image/jpeg', 100, 100),
	(11, 'f2', 'frame2.jpg', 'images/f2', 'image/jpeg', 100, 100),
	(12, 'f3', 'frame3.jpg', 'images/f3', 'image/jpeg', 100, 100)
`); err != nil {
		t.Fatalf("seed images: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms)
VALUES
	(1, 10, 0, 1000),
	(1, 11, 1, 7000),
	(2, 12, 0, 500)
`); err != nil {
		t.Fatalf("seed video frames: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO index_jobs(kind, image_id, model_id, state)
VALUES
	('embed_image', 10, 1, 'done'),
	('embed_image', 11, 1, 'pending'),
	('embed_image', 12, 1, 'failed')
`); err != nil {
		t.Fatalf("seed jobs: %v", err)
	}

	return dbConn
}

func TestListVideosReturnsResults(t *testing.T) {
	dbConn := setupVideosDB(t)
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})

	req := httptest.NewRequest(http.MethodGet, "/api/videos?limit=10&offset=0", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var resp ListResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Total != 2 || len(resp.Videos) != 2 {
		t.Fatalf("unexpected video response: %+v", resp)
	}
	if resp.Videos[0].VideoID != 2 || resp.Videos[1].VideoID != 1 {
		t.Fatalf("unexpected order: %+v", resp.Videos)
	}
	if resp.Videos[0].IndexState != "failed" {
		t.Fatalf("expected failed state for video 2, got %+v", resp.Videos[0])
	}
	if resp.Videos[1].IndexState != "pending" {
		t.Fatalf("expected pending state for video 1, got %+v", resp.Videos[1])
	}
	if resp.Videos[1].PreviewPath != "images/f1" || resp.Videos[1].ImageID != 10 {
		t.Fatalf("unexpected preview frame for video 1: %+v", resp.Videos[1])
	}
	if resp.Videos[1].TranscriptText != "tis better to remain silent" {
		t.Fatalf("expected transcript text in video list, got %+v", resp.Videos[1])
	}
}

func TestListVideosRejectsInvalidMethod(t *testing.T) {
	dbConn := setupVideosDB(t)
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})

	req := httptest.NewRequest(http.MethodPost, "/api/videos", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status: got=%d want=%d", rr.Code, http.StatusMethodNotAllowed)
	}
}
