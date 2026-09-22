package mediastatus

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
)

func setup(t *testing.T) *sql.DB {
	t.Helper()
	conn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	if err := db.RunMigrations(context.Background(), conn); err != nil {
		t.Fatal(err)
	}
	if _, err := conn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height, description, annotation_updated_at, reannotate_requested) VALUES
  (1, 'a', 'a.jpg', 'images/a', 'image/jpeg', 1, 1, 'Described.', '2026-09-22 01:00:00', 0),
  (2, 'b', 'b.jpg', 'images/b', 'image/jpeg', 1, 1, '', NULL, 0),
  (3, 'c', 'c.jpg', 'images/c', 'image/jpeg', 1, 1, 'Old text.', '2026-09-22 00:00:00', 1),
  (4, 'd', 'd.jpg', 'images/d', 'image/jpeg', 1, 1, '', NULL, 0);
INSERT INTO index_jobs(kind, image_id, model_id, state) VALUES
  ('embed_image', 1, 1, 'done'), ('annotate_image', 1, 1, 'done'),
  ('embed_image', 2, 1, 'done'), ('annotate_image', 2, 1, 'leased'),
  ('embed_image', 3, 1, 'done'), ('annotate_image', 3, 1, 'pending'),
  ('embed_image', 4, 1, 'failed'), ('annotate_image', 4, 1, 'failed');
INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count, description, annotation_updated_at) VALUES
  (1, 'v', 'v.mp4', 'videos/v', 'video/mp4', 1000, 1, 1, 1, 'Video text.', '2026-09-22 02:00:00');
INSERT INTO index_jobs(kind, video_id, model_id, state) VALUES ('annotate_video', 1, 1, 'leased');
`); err != nil {
		t.Fatalf("seed: %v", err)
	}
	return conn
}

func TestStatusReportsAnnotationStates(t *testing.T) {
	conn := setup(t)
	h := NewHandler(&Handler{DB: conn, ModelID: 1})
	req := httptest.NewRequest(http.MethodGet, "/api/media/status?images=1,2,3,4,99,x&videos=1", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status: %d %s", rr.Code, rr.Body.String())
	}
	var resp Response
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	want := map[int64]string{1: "done", 2: "annotating", 3: "queued", 4: "failed"}
	if len(resp.Images) != 4 {
		t.Fatalf("images: %+v", resp.Images)
	}
	for _, s := range resp.Images {
		if s.AnnotationState != want[s.ImageID] {
			t.Fatalf("image %d: got %q want %q", s.ImageID, s.AnnotationState, want[s.ImageID])
		}
		if s.ImageID == 4 && s.IndexState != "failed" {
			t.Fatalf("image 4 index state: %q", s.IndexState)
		}
		if s.ImageID == 1 && s.AnnotationUpdatedAt != "2026-09-22 01:00:00" {
			t.Fatalf("image 1 updated at: %q", s.AnnotationUpdatedAt)
		}
	}
	if len(resp.Videos) != 1 || resp.Videos[0].AnnotationState != "annotating" || resp.Videos[0].AnnotationUpdatedAt != "2026-09-22 02:00:00" {
		t.Fatalf("videos: %+v", resp.Videos)
	}
}

func TestStatusRejectsTooManyIDsAndWrongMethod(t *testing.T) {
	conn := setup(t)
	h := NewHandler(&Handler{DB: conn, ModelID: 1})
	ids := make([]string, MaxIDsPerRequest+1)
	for i := range ids {
		ids[i] = strconv.Itoa(i + 1)
	}
	req := httptest.NewRequest(http.MethodGet, "/api/media/status?images="+strings.Join(ids, ","), nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("too many ids: %d", rr.Code)
	}
	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/api/media/status", nil))
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("post: %d", rr.Code)
	}
	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/api/media/status", nil))
	if rr.Code != http.StatusOK || strings.TrimSpace(rr.Body.String()) != `{"images":[],"videos":[]}` {
		t.Fatalf("empty: %d %s", rr.Code, rr.Body.String())
	}
}
