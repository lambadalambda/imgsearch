package videos

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
	"imgsearch/internal/vectorindex"
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
UPDATE videos
SET description = 'A vocalist performs on stage while a crowd gathers near the front.',
    tags_json = '["concert","music","stage"]'
WHERE id = 1
`); err != nil {
		t.Fatalf("seed video annotations: %v", err)
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
	if resp.Videos[1].Description == "" {
		t.Fatalf("expected video description in video list, got %+v", resp.Videos[1])
	}
	if len(resp.Videos[1].Tags) != 3 || resp.Videos[1].Tags[0] != "concert" {
		t.Fatalf("expected video tags in video list, got %+v", resp.Videos[1])
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

func TestListVideosNSFWFiltering(t *testing.T) {
	dbConn := setupVideosDB(t)
	if _, err := dbConn.Exec(`
UPDATE images
SET tags_json = '["clip","nsfw"]'
WHERE id = 12
`); err != nil {
		t.Fatalf("seed nsfw frame tags: %v", err)
	}

	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})

	defaultReq := httptest.NewRequest(http.MethodGet, "/api/videos?limit=10&offset=0", nil)
	defaultRR := httptest.NewRecorder()
	h.ServeHTTP(defaultRR, defaultReq)

	if defaultRR.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", defaultRR.Code, defaultRR.Body.String())
	}

	var defaultResp ListResponse
	if err := json.Unmarshal(defaultRR.Body.Bytes(), &defaultResp); err != nil {
		t.Fatalf("decode default response: %v", err)
	}
	if defaultResp.Total != 1 {
		t.Fatalf("expected total=1 with nsfw hidden by default, got %d", defaultResp.Total)
	}
	if len(defaultResp.Videos) != 1 || defaultResp.Videos[0].VideoID != 1 {
		t.Fatalf("expected only non-nsfw video in default list, got %+v", defaultResp.Videos)
	}

	includeReq := httptest.NewRequest(http.MethodGet, "/api/videos?limit=10&offset=0&include_nsfw=1", nil)
	includeRR := httptest.NewRecorder()
	h.ServeHTTP(includeRR, includeReq)

	if includeRR.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", includeRR.Code, includeRR.Body.String())
	}

	var includeResp ListResponse
	if err := json.Unmarshal(includeRR.Body.Bytes(), &includeResp); err != nil {
		t.Fatalf("decode include response: %v", err)
	}
	if includeResp.Total != 2 {
		t.Fatalf("expected total=2 with include_nsfw=1, got %d", includeResp.Total)
	}
}

func TestDeleteVideoRemovesVideoFramesTranscriptAndOrphanFiles(t *testing.T) {
	dbConn := setupVideosDB(t)
	dataDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dataDir, "videos"), 0o755); err != nil {
		t.Fatalf("mkdir videos dir: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(dataDir, "images"), 0o755); err != nil {
		t.Fatalf("mkdir images dir: %v", err)
	}
	for _, rel := range []string{"videos/v1", "images/f1", "images/f2"} {
		if err := os.WriteFile(filepath.Join(dataDir, filepath.FromSlash(rel)), []byte(rel), 0o644); err != nil {
			t.Fatalf("write fixture file %s: %v", rel, err)
		}
	}
	if _, err := dbConn.Exec(`
INSERT INTO video_transcript_embeddings(video_id, model_id, dim, vector_blob)
VALUES (1, 1, 2, ?)
`, vectorindex.FloatsToBlob([]float32{1, 2})); err != nil {
		t.Fatalf("seed transcript embedding: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO image_embeddings(image_id, model_id, dim, vector_blob)
VALUES
	(10, 1, 2, ?),
	(11, 1, 2, ?)
`, vectorindex.FloatsToBlob([]float32{1, 2}), vectorindex.FloatsToBlob([]float32{3, 4})); err != nil {
		t.Fatalf("seed image embeddings: %v", err)
	}

	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: dataDir})
	req := httptest.NewRequest(http.MethodDelete, "/api/videos/1", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusNoContent {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	assertCount(t, dbConn, `SELECT COUNT(*) FROM videos WHERE id = 1`, 0)
	assertCount(t, dbConn, `SELECT COUNT(*) FROM video_frames WHERE video_id = 1`, 0)
	assertCount(t, dbConn, `SELECT COUNT(*) FROM video_transcript_embeddings WHERE video_id = 1`, 0)
	assertCount(t, dbConn, `SELECT COUNT(*) FROM images WHERE id IN (10,11)`, 0)
	assertCount(t, dbConn, `SELECT COUNT(*) FROM image_embeddings WHERE image_id IN (10,11)`, 0)
	assertCount(t, dbConn, `SELECT COUNT(*) FROM index_jobs WHERE video_id = 1 OR image_id IN (10,11)`, 0)
	for _, rel := range []string{"videos/v1", "images/f1", "images/f2"} {
		if _, err := os.Stat(filepath.Join(dataDir, filepath.FromSlash(rel))); !os.IsNotExist(err) {
			t.Fatalf("expected file %s removed, stat err=%v", rel, err)
		}
	}
}

func TestDeleteVideoKeepsSharedFrameImageUsedByOtherVideo(t *testing.T) {
	dbConn := setupVideosDB(t)
	dataDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dataDir, "videos"), 0o755); err != nil {
		t.Fatalf("mkdir videos dir: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(dataDir, "images"), 0o755); err != nil {
		t.Fatalf("mkdir images dir: %v", err)
	}
	for _, rel := range []string{"videos/v1", "videos/v3", "images/f1"} {
		if err := os.WriteFile(filepath.Join(dataDir, filepath.FromSlash(rel)), []byte(rel), 0o644); err != nil {
			t.Fatalf("write fixture file %s: %v", rel, err)
		}
	}
	if _, err := dbConn.Exec(`
INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count)
VALUES (3, 'v3', 'third.mp4', 'videos/v3', 'video/mp4', 1000, 320, 240, 1)
`); err != nil {
		t.Fatalf("seed extra video: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms)
VALUES (3, 10, 0, 100)
`); err != nil {
		t.Fatalf("seed shared frame: %v", err)
	}

	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: dataDir})
	req := httptest.NewRequest(http.MethodDelete, "/api/videos/1", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusNoContent {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	assertCount(t, dbConn, `SELECT COUNT(*) FROM videos WHERE id = 3`, 1)
	assertCount(t, dbConn, `SELECT COUNT(*) FROM images WHERE id = 10`, 1)
	if _, err := os.Stat(filepath.Join(dataDir, "images", "f1")); err != nil {
		t.Fatalf("expected shared frame file retained: %v", err)
	}
}

func assertCount(t *testing.T, dbConn *sql.DB, query string, want int) {
	t.Helper()
	var got int
	if err := dbConn.QueryRow(query).Scan(&got); err != nil {
		t.Fatalf("query count %q: %v", query, err)
	}
	if got != want {
		t.Fatalf("count for %q: got=%d want=%d", query, got, want)
	}
}
