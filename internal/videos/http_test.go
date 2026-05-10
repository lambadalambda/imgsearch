package videos

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
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
	(10, 'f1', 'frame1.jpg', 'images/f1', 'image/jpeg', 1080, 1920),
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
	if resp.Videos[1].PreviewWidth != 1080 || resp.Videos[1].PreviewHeight != 1920 {
		t.Fatalf("expected preview frame dimensions for video 1, got %+v", resp.Videos[1])
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

func TestListVideosRandomOrderIsStableAcrossPages(t *testing.T) {
	dbConn := setupVideosDB(t)
	for id := int64(3); id <= 12; id++ {
		if _, err := dbConn.Exec(`
INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count)
VALUES (?, ?, ?, ?, 'video/mp4', 1000, 640, 480, 0)
`, id, fmt.Sprintf("v-extra-%d", id), fmt.Sprintf("extra-%d.mp4", id), fmt.Sprintf("videos/extra-%d", id)); err != nil {
			t.Fatalf("seed extra video %d: %v", id, err)
		}
	}
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})

	firstPage := videoIDsForRequest(t, h, "/api/videos?limit=5&offset=0&order=random&seed=7")
	secondPage := videoIDsForRequest(t, h, "/api/videos?limit=7&offset=5&order=random&seed=7")
	fullPage := videoIDsForRequest(t, h, "/api/videos?limit=20&offset=0&order=random&seed=7")
	otherSeedPage := videoIDsForRequest(t, h, "/api/videos?limit=20&offset=0&order=random&seed=1200000000")

	paged := append(append([]int64{}, firstPage...), secondPage...)
	if !reflect.DeepEqual(paged, fullPage) {
		t.Fatalf("expected seeded random pages to match full order: pages=%v full=%v", paged, fullPage)
	}
	if reflect.DeepEqual(fullPage, []int64{12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}) {
		t.Fatalf("expected seeded random order, got newest-first order %v", fullPage)
	}
	if isInt64Rotation(fullPage, otherSeedPage) {
		t.Fatalf("expected different seeds to produce more than a rotated order: seed7=%v other=%v", fullPage, otherSeedPage)
	}
	if len(fullPage) != 12 {
		t.Fatalf("expected both videos without duplicates, got %v", fullPage)
	}
	seen := map[int64]bool{}
	for _, id := range fullPage {
		if seen[id] {
			t.Fatalf("expected no duplicate videos in seeded random order, got %v", fullPage)
		}
		seen[id] = true
	}
}

func isInt64Rotation(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	if len(a) == 0 {
		return true
	}
	for offset := range a {
		matched := true
		for i := range a {
			if a[(i+offset)%len(a)] != b[i] {
				matched = false
				break
			}
		}
		if matched {
			return true
		}
	}
	return false
}

func videoIDsForRequest(t *testing.T, h http.Handler, path string) []int64 {
	t.Helper()

	req := httptest.NewRequest(http.MethodGet, path, nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status for %s: got=%d body=%s", path, rr.Code, rr.Body.String())
	}

	var resp ListResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response for %s: %v", path, err)
	}
	ids := make([]int64, 0, len(resp.Videos))
	for _, item := range resp.Videos {
		ids = append(ids, item.VideoID)
	}
	return ids
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
	if rr.Header().Get("Allow") != http.MethodGet {
		t.Fatalf("allow: got=%q want=%q", rr.Header().Get("Allow"), http.MethodGet)
	}
}

func TestListVideosRequiresExactCollectionPath(t *testing.T) {
	dbConn := setupVideosDB(t)
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})

	req := httptest.NewRequest(http.MethodGet, "/api/videos/1", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusNotFound {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusNotFound, rr.Body.String())
	}
}

func TestVideosHandlerRejectsMissingDependencies(t *testing.T) {
	h := NewHandler(nil)

	req := httptest.NewRequest(http.MethodGet, "/api/videos", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusServiceUnavailable, rr.Body.String())
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

func TestReannotateVideoCreatesAnnotationJob(t *testing.T) {
	dbConn := setupVideosDB(t)
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})

	req := httptest.NewRequest(http.MethodPost, "/api/videos/2/reannotate", nil)
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
JOIN videos ON videos.id = index_jobs.video_id
WHERE kind = 'annotate_video' AND video_id = 2 AND model_id = 1
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
		t.Fatalf("expected video reannotate_requested=1, got %d", reannotateRequested)
	}
}

func TestReannotateVideoResetsCompletedJobToPending(t *testing.T) {
	dbConn := setupVideosDB(t)
	if _, err := dbConn.Exec(`
UPDATE videos
SET annotation_updated_at = datetime('now')
WHERE id = 1
`); err != nil {
		t.Fatalf("seed annotation timestamp: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO index_jobs(kind, image_id, video_id, model_id, state, attempts, run_after, leased_until, lease_owner, last_error)
VALUES ('annotate_video', NULL, 1, 1, 'done', 2, datetime('now', '+2 minutes'), datetime('now', '+3 minutes'), 'worker-1', 'timeout')
`); err != nil {
		t.Fatalf("seed done annotate job: %v", err)
	}

	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})
	req := httptest.NewRequest(http.MethodPost, "/api/videos/1/reannotate", nil)
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
JOIN videos ON videos.id = index_jobs.video_id
WHERE kind = 'annotate_video' AND video_id = 1 AND model_id = 1
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
		t.Fatalf("expected video reannotate_requested=1, got %d", reannotateRequested)
	}

	var description string
	var tagsJSON string
	var annotationUpdatedAt sql.NullString
	if err := dbConn.QueryRow(`
SELECT description, tags_json, annotation_updated_at
FROM videos
WHERE id = 1
`).Scan(&description, &tagsJSON, &annotationUpdatedAt); err != nil {
		t.Fatalf("load video annotations: %v", err)
	}
	if description != "" {
		t.Fatalf("expected video description cleared, got %q", description)
	}
	if tagsJSON != "[]" {
		t.Fatalf("expected video tags_json reset to [], got %q", tagsJSON)
	}
	if annotationUpdatedAt.Valid {
		t.Fatalf("expected annotation_updated_at cleared, got %q", annotationUpdatedAt.String)
	}
}

func TestToggleVideoNSFWTag(t *testing.T) {
	dbConn := setupVideosDB(t)
	if _, err := dbConn.Exec(`
UPDATE videos
SET tags_json = '["concert","music"]'
WHERE id = 1
`); err != nil {
		t.Fatalf("seed video tags: %v", err)
	}
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1})

	firstReq := httptest.NewRequest(http.MethodPost, "/api/videos/1/toggle-nsfw", nil)
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
FROM videos
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

	secondReq := httptest.NewRequest(http.MethodPost, "/api/videos/1/toggle-nsfw", nil)
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
FROM videos
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
