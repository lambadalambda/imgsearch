package images

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
	"imgsearch/internal/vectorindex"
)

func setupImagesDB(t *testing.T) *sql.DB {
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
	(2, 'b', 'two.jpg', 'images/b', 'image/jpeg', 20, 20),
	(3, 'c', 'three.jpg', 'images/c', 'image/jpeg', 30, 30)
`)
	if err != nil {
		t.Fatalf("seed images: %v", err)
	}

	_, err = dbConn.Exec(`
INSERT INTO index_jobs(kind, image_id, model_id, state)
VALUES
	('embed_image', 1, 1, 'done'),
	('embed_image', 2, 1, 'pending'),
	('embed_image', 3, 1, 'failed')
`)
	if err != nil {
		t.Fatalf("seed jobs: %v", err)
	}

	return dbConn
}

func TestListImagesReturnsResults(t *testing.T) {
	dbConn := setupImagesDB(t)
	if _, err := dbConn.Exec(`
UPDATE images
SET description = 'A stored gallery description.', tags_json = '["gallery","sample"]'
WHERE id = 3
`); err != nil {
		t.Fatalf("seed annotations: %v", err)
	}
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})

	req := httptest.NewRequest(http.MethodGet, "/api/images?limit=2&offset=0", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var resp ListResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Images) != 2 {
		t.Fatalf("expected 2 images, got %d", len(resp.Images))
	}
	if resp.Total != 3 {
		t.Fatalf("expected total=3, got %d", resp.Total)
	}
	if resp.Images[0].ImageID != 3 || resp.Images[1].ImageID != 2 {
		t.Fatalf("unexpected order: %+v", resp.Images)
	}
	if resp.Images[0].IndexState != "failed" {
		t.Fatalf("expected failed state, got %s", resp.Images[0].IndexState)
	}
	if resp.Images[1].IndexState != "pending" {
		t.Fatalf("expected pending state, got %s", resp.Images[1].IndexState)
	}
	if resp.Images[0].Description != "A stored gallery description." {
		t.Fatalf("unexpected description: %q", resp.Images[0].Description)
	}
	if len(resp.Images[0].Tags) != 2 || resp.Images[0].Tags[0] != "gallery" {
		t.Fatalf("unexpected tags: %v", resp.Images[0].Tags)
	}
}

func TestListImagesRejectsInvalidMethod(t *testing.T) {
	dbConn := setupImagesDB(t)
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})

	req := httptest.NewRequest(http.MethodPost, "/api/images", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status: got=%d want=%d", rr.Code, http.StatusMethodNotAllowed)
	}
}

func TestListImagesUsesFallbacksForBadPagination(t *testing.T) {
	dbConn := setupImagesDB(t)
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})

	req := httptest.NewRequest(http.MethodGet, "/api/images?limit=-1&offset=nope", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var resp ListResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Images) != 3 {
		t.Fatalf("expected 3 images with default limit, got %d", len(resp.Images))
	}
}

func TestListImagesDefaultsToPendingWhenNoJobExists(t *testing.T) {
	dbConn := setupImagesDB(t)

	_, err := dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES (99, 'z', 'nojob.jpg', 'images/z', 'image/jpeg', 9, 9)
`)
	if err != nil {
		t.Fatalf("insert image: %v", err)
	}

	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})
	req := httptest.NewRequest(http.MethodGet, "/api/images?limit=1&offset=0", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var resp ListResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Images) != 1 {
		t.Fatalf("expected 1 image, got %d", len(resp.Images))
	}
	if resp.Images[0].ImageID != 99 {
		t.Fatalf("expected image 99 first, got %d", resp.Images[0].ImageID)
	}
	if resp.Images[0].IndexState != "pending" {
		t.Fatalf("expected pending state fallback, got %s", resp.Images[0].IndexState)
	}
}

func TestListImagesDoesNotRequeueDoneJobsMissingAnnotations(t *testing.T) {
	dbConn := setupImagesDB(t)

	if _, err := dbConn.Exec(`
UPDATE index_jobs
SET attempts = 2,
    max_attempts = 3,
    last_error = 'annotation timeout'
WHERE kind = 'embed_image' AND model_id = 1 AND image_id = 1
`); err != nil {
		t.Fatalf("seed job retry metadata: %v", err)
	}

	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})
	req := httptest.NewRequest(http.MethodGet, "/api/images?limit=3&offset=0", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var resp ListResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Images) != 3 {
		t.Fatalf("expected 3 images, got %d", len(resp.Images))
	}
	if resp.Images[2].ImageID != 1 {
		t.Fatalf("expected image 1 in third slot, got id=%d", resp.Images[2].ImageID)
	}
	if resp.Images[2].IndexState != "done" {
		t.Fatalf("expected done state to remain visible, got %s", resp.Images[2].IndexState)
	}

	var state string
	var attempts int
	var lastError string
	if err := dbConn.QueryRow(`
SELECT state, attempts, COALESCE(last_error, '')
FROM index_jobs
WHERE kind = 'embed_image' AND model_id = 1 AND image_id = 1
`).Scan(&state, &attempts, &lastError); err != nil {
		t.Fatalf("load job after list: %v", err)
	}
	if state != "done" {
		t.Fatalf("expected job state to remain done, got %s", state)
	}
	if attempts != 2 {
		t.Fatalf("expected attempts to remain 2, got %d", attempts)
	}
	if lastError != "annotation timeout" {
		t.Fatalf("expected last_error to remain unchanged, got %q", lastError)
	}
}

func TestListImagesExcludesDerivedVideoFrames(t *testing.T) {
	dbConn := setupImagesDB(t)
	if _, err := dbConn.Exec(`
INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count)
VALUES (9, 'vid', 'clip.mp4', 'videos/vid', 'video/mp4', 12000, 1920, 1080, 1)
`); err != nil {
		t.Fatalf("seed video: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms)
VALUES (9, 3, 0, 500)
`); err != nil {
		t.Fatalf("seed video frame: %v", err)
	}

	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})
	req := httptest.NewRequest(http.MethodGet, "/api/images?limit=10&offset=0", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var resp ListResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Total != 2 {
		t.Fatalf("expected total=2 after excluding video frames, got %d", resp.Total)
	}
	for _, item := range resp.Images {
		if item.ImageID == 3 {
			t.Fatalf("expected derived frame image 3 to be excluded, got %+v", resp.Images)
		}
	}
}

func TestDeleteImageRemovesRowJobsEmbeddingsAndFile(t *testing.T) {
	dbConn := setupImagesDB(t)
	dataDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dataDir, "images"), 0o755); err != nil {
		t.Fatalf("mkdir images dir: %v", err)
	}
	imagePath := filepath.Join(dataDir, "images", "a")
	if err := os.WriteFile(imagePath, []byte("image"), 0o644); err != nil {
		t.Fatalf("write image file: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO image_embeddings(image_id, model_id, dim, vector_blob)
VALUES (1, 1, 2, ?)
`, vectorindex.FloatsToBlob([]float32{1, 2})); err != nil {
		t.Fatalf("seed image embedding: %v", err)
	}

	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: dataDir})
	req := httptest.NewRequest(http.MethodDelete, "/api/images/1", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusNoContent {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	assertMissingRow(t, dbConn, `SELECT COUNT(*) FROM images WHERE id = 1`)
	assertMissingRow(t, dbConn, `SELECT COUNT(*) FROM image_embeddings WHERE image_id = 1`)
	assertMissingRow(t, dbConn, `SELECT COUNT(*) FROM index_jobs WHERE image_id = 1`)
	if _, err := os.Stat(imagePath); !os.IsNotExist(err) {
		t.Fatalf("expected image file removed, stat err=%v", err)
	}
}

func TestDeleteImageRejectsDerivedVideoFrame(t *testing.T) {
	dbConn := setupImagesDB(t)
	if _, err := dbConn.Exec(`
INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count)
VALUES (9, 'vid', 'clip.mp4', 'videos/vid', 'video/mp4', 12000, 1920, 1080, 1)
`); err != nil {
		t.Fatalf("seed video: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms)
VALUES (9, 3, 0, 500)
`); err != nil {
		t.Fatalf("seed video frame: %v", err)
	}

	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: t.TempDir()})
	req := httptest.NewRequest(http.MethodDelete, "/api/images/3", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusConflict {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}
}

func assertMissingRow(t *testing.T, dbConn *sql.DB, query string, args ...any) {
	t.Helper()
	var count int
	if err := dbConn.QueryRow(query, args...).Scan(&count); err != nil {
		t.Fatalf("count rows with %q: %v", query, err)
	}
	if count != 0 {
		t.Fatalf("expected 0 rows for %s, got %d", fmt.Sprintf(query, args...), count)
	}
}
