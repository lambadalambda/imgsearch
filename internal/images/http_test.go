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
	if rr.Header().Get("Allow") != http.MethodGet {
		t.Fatalf("allow: got=%q want=%q", rr.Header().Get("Allow"), http.MethodGet)
	}
}

func TestListImagesRequiresExactCollectionPath(t *testing.T) {
	dbConn := setupImagesDB(t)
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})

	req := httptest.NewRequest(http.MethodGet, "/api/images/1", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusNotFound {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusNotFound, rr.Body.String())
	}
}

func TestImagesHandlerRejectsMissingDependencies(t *testing.T) {
	h := NewHandler(nil)

	req := httptest.NewRequest(http.MethodGet, "/api/images", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusServiceUnavailable, rr.Body.String())
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

func TestListImagesNSFWFiltering(t *testing.T) {
	dbConn := setupImagesDB(t)
	if _, err := dbConn.Exec(`
UPDATE images
SET tags_json = '["portrait","nsfw"]'
WHERE id = 2
`); err != nil {
		t.Fatalf("seed nsfw tags: %v", err)
	}

	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})

	defaultReq := httptest.NewRequest(http.MethodGet, "/api/images?limit=10&offset=0", nil)
	defaultRR := httptest.NewRecorder()
	h.ServeHTTP(defaultRR, defaultReq)

	if defaultRR.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", defaultRR.Code, defaultRR.Body.String())
	}

	var defaultResp ListResponse
	if err := json.Unmarshal(defaultRR.Body.Bytes(), &defaultResp); err != nil {
		t.Fatalf("decode default response: %v", err)
	}
	if defaultResp.Total != 2 {
		t.Fatalf("expected total=2 with nsfw hidden by default, got %d", defaultResp.Total)
	}
	for _, item := range defaultResp.Images {
		if item.ImageID == 2 {
			t.Fatalf("expected nsfw image 2 excluded by default, got %+v", defaultResp.Images)
		}
	}

	includeReq := httptest.NewRequest(http.MethodGet, "/api/images?limit=10&offset=0&include_nsfw=1", nil)
	includeRR := httptest.NewRecorder()
	h.ServeHTTP(includeRR, includeReq)

	if includeRR.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", includeRR.Code, includeRR.Body.String())
	}

	var includeResp ListResponse
	if err := json.Unmarshal(includeRR.Body.Bytes(), &includeResp); err != nil {
		t.Fatalf("decode include response: %v", err)
	}
	if includeResp.Total != 3 {
		t.Fatalf("expected total=3 with include_nsfw=1, got %d", includeResp.Total)
	}
	foundNSFW := false
	for _, item := range includeResp.Images {
		if item.ImageID == 2 {
			foundNSFW = true
			break
		}
	}
	if !foundNSFW {
		t.Fatalf("expected nsfw image 2 returned with include_nsfw=1, got %+v", includeResp.Images)
	}
}

func TestReannotateImageCreatesAnnotationJob(t *testing.T) {
	dbConn := setupImagesDB(t)
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})

	req := httptest.NewRequest(http.MethodPost, "/api/images/2/reannotate", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusAccepted {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var state string
	var attempts int
	var lastError sql.NullString
	var reannotateRequested int
	if err := dbConn.QueryRow(`
SELECT state, attempts, last_error, reannotate_requested
FROM index_jobs
JOIN images ON images.id = index_jobs.image_id
WHERE kind = 'annotate_image' AND image_id = 2 AND model_id = 1
`).Scan(&state, &attempts, &lastError, &reannotateRequested); err != nil {
		t.Fatalf("load annotate job: %v", err)
	}
	if state != "pending" {
		t.Fatalf("expected pending state, got %s", state)
	}
	if attempts != 0 {
		t.Fatalf("expected attempts reset to 0, got %d", attempts)
	}
	if lastError.Valid {
		t.Fatalf("expected last_error cleared, got %q", lastError.String)
	}
	if reannotateRequested != 1 {
		t.Fatalf("expected image reannotate_requested=1, got %d", reannotateRequested)
	}
}

func TestReannotateImageResetsCompletedJobToPending(t *testing.T) {
	dbConn := setupImagesDB(t)
	if _, err := dbConn.Exec(`
UPDATE images
SET description = 'old annotation', tags_json = '["old","tags"]'
WHERE id = 1
`); err != nil {
		t.Fatalf("seed image annotations: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO index_jobs(kind, image_id, model_id, state, attempts, run_after, leased_until, lease_owner, last_error)
VALUES ('annotate_image', 1, 1, 'done', 2, datetime('now', '+2 minutes'), datetime('now', '+3 minutes'), 'worker-1', 'timeout')
`); err != nil {
		t.Fatalf("seed done annotate job: %v", err)
	}

	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})
	req := httptest.NewRequest(http.MethodPost, "/api/images/1/reannotate", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusAccepted {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var state string
	var attempts int
	var runAfter sql.NullString
	var leasedUntil sql.NullString
	var leaseOwner sql.NullString
	var lastError sql.NullString
	var reannotateRequested int
	if err := dbConn.QueryRow(`
SELECT state, attempts, run_after, leased_until, lease_owner, last_error, reannotate_requested
FROM index_jobs
JOIN images ON images.id = index_jobs.image_id
WHERE kind = 'annotate_image' AND image_id = 1 AND model_id = 1
`).Scan(&state, &attempts, &runAfter, &leasedUntil, &leaseOwner, &lastError, &reannotateRequested); err != nil {
		t.Fatalf("load annotate job: %v", err)
	}
	if state != "pending" {
		t.Fatalf("expected pending state, got %s", state)
	}
	if attempts != 0 {
		t.Fatalf("expected attempts reset to 0, got %d", attempts)
	}
	if runAfter.Valid {
		t.Fatalf("expected run_after cleared, got %q", runAfter.String)
	}
	if leasedUntil.Valid {
		t.Fatalf("expected leased_until cleared, got %q", leasedUntil.String)
	}
	if leaseOwner.Valid {
		t.Fatalf("expected lease_owner cleared, got %q", leaseOwner.String)
	}
	if lastError.Valid {
		t.Fatalf("expected last_error cleared, got %q", lastError.String)
	}
	if reannotateRequested != 1 {
		t.Fatalf("expected image reannotate_requested=1, got %d", reannotateRequested)
	}

	var description string
	var tagsJSON string
	if err := dbConn.QueryRow(`
SELECT description, tags_json
FROM images
WHERE id = 1
`).Scan(&description, &tagsJSON); err != nil {
		t.Fatalf("load image annotations: %v", err)
	}
	if description != "" {
		t.Fatalf("expected image description cleared, got %q", description)
	}
	if tagsJSON != "[]" {
		t.Fatalf("expected image tags_json reset to [], got %q", tagsJSON)
	}
}

func TestToggleImageNSFWTag(t *testing.T) {
	dbConn := setupImagesDB(t)
	if _, err := dbConn.Exec(`
UPDATE images
SET tags_json = '["portrait"]'
WHERE id = 1
`); err != nil {
		t.Fatalf("seed image tags: %v", err)
	}
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})

	firstReq := httptest.NewRequest(http.MethodPost, "/api/images/1/toggle-nsfw", nil)
	firstRR := httptest.NewRecorder()
	h.ServeHTTP(firstRR, firstReq)

	if firstRR.Code != http.StatusOK {
		t.Fatalf("first toggle status: got=%d body=%s", firstRR.Code, firstRR.Body.String())
	}

	var firstPayload struct {
		IsNSFW bool `json:"is_nsfw"`
	}
	if err := json.Unmarshal(firstRR.Body.Bytes(), &firstPayload); err != nil {
		t.Fatalf("decode first toggle response: %v", err)
	}
	if !firstPayload.IsNSFW {
		t.Fatalf("expected first toggle to enable nsfw, got %+v", firstPayload)
	}

	var firstTagsJSON string
	if err := dbConn.QueryRow(`
SELECT COALESCE(tags_json, '[]')
FROM images
WHERE id = 1
`).Scan(&firstTagsJSON); err != nil {
		t.Fatalf("load first tags_json: %v", err)
	}
	firstTags, err := decodeTags(firstTagsJSON)
	if err != nil {
		t.Fatalf("decode first tags_json: %v", err)
	}
	if !hasTag(firstTags, "nsfw") {
		t.Fatalf("expected nsfw tag after first toggle, got %v", firstTags)
	}

	secondReq := httptest.NewRequest(http.MethodPost, "/api/images/1/toggle-nsfw", nil)
	secondRR := httptest.NewRecorder()
	h.ServeHTTP(secondRR, secondReq)

	if secondRR.Code != http.StatusOK {
		t.Fatalf("second toggle status: got=%d body=%s", secondRR.Code, secondRR.Body.String())
	}

	var secondPayload struct {
		IsNSFW bool `json:"is_nsfw"`
	}
	if err := json.Unmarshal(secondRR.Body.Bytes(), &secondPayload); err != nil {
		t.Fatalf("decode second toggle response: %v", err)
	}
	if secondPayload.IsNSFW {
		t.Fatalf("expected second toggle to disable nsfw, got %+v", secondPayload)
	}

	var secondTagsJSON string
	if err := dbConn.QueryRow(`
SELECT COALESCE(tags_json, '[]')
FROM images
WHERE id = 1
`).Scan(&secondTagsJSON); err != nil {
		t.Fatalf("load second tags_json: %v", err)
	}
	secondTags, err := decodeTags(secondTagsJSON)
	if err != nil {
		t.Fatalf("decode second tags_json: %v", err)
	}
	if hasTag(secondTags, "nsfw") {
		t.Fatalf("expected nsfw tag removed after second toggle, got %v", secondTags)
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

func decodeTags(raw string) ([]string, error) {
	if raw == "" {
		return nil, nil
	}
	var tags []string
	if err := json.Unmarshal([]byte(raw), &tags); err != nil {
		return nil, err
	}
	return tags, nil
}

func hasTag(tags []string, target string) bool {
	for _, tag := range tags {
		if tag == target {
			return true
		}
	}
	return false
}
