package images

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
