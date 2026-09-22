//go:build cgo

package app

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"image"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	_ "golang.org/x/image/webp"
	"imgsearch/internal/httputil"
	"imgsearch/internal/images"
	"imgsearch/internal/vectorindex/bruteforce"
)

func TestRuntimeThumbnailUploadServeAndDelete(t *testing.T) {
	conn := openDB(t)
	if err := Bootstrap(context.Background(), conn, func(context.Context, *sql.DB) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if _, err := conn.Exec(`INSERT INTO embedding_models(id, name, version, dimensions, metric, normalized) VALUES(1, 'test', 'v1', 2, 'cosine', 1)`); err != nil {
		t.Fatal(err)
	}
	data := &DataRuntime{DB: conn, Index: bruteforce.NewIndex(conn), VectorBackend: "bruteforce"}
	dir := t.TempDir()
	runtime, err := NewRuntime(RuntimeOptions{Data: data, DataDir: dir, ModelID: 1, Embedder: fakeRuntimeEmbedder{}, Index: data.Index})
	if err != nil {
		t.Fatal(err)
	}
	original, err := os.ReadFile("../../fixtures/images/cat_1.jpg")
	if err != nil {
		t.Fatal(err)
	}
	uploaded, err := runtime.UploadService.Store(context.Background(), "cat.jpg", bytes.NewReader(original))
	if err != nil {
		t.Fatal(err)
	}
	protected := httputil.NewAPIAuthMiddleware("test-secret")(runtime.Mux)
	get := func(method, url string, authenticated bool) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, url, nil)
		if authenticated {
			req.Header.Set(httputil.APIKeyHeaderName, "test-secret")
		}
		rr := httptest.NewRecorder()
		protected.ServeHTTP(rr, req)
		return rr
	}
	rr := get("GET", "/api/images", true)
	var page images.ListResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &page); err != nil {
		t.Fatal(err)
	}
	if len(page.Images) != 1 {
		t.Fatalf("images=%s", rr.Body)
	}
	item := page.Images[0]
	url := "/media/" + item.ThumbnailPath
	if rr := get("GET", url, false); rr.Code != 401 {
		t.Fatalf("anonymous status=%d", rr.Code)
	}
	thumb := get("GET", url, true)
	if thumb.Code != 200 || thumb.Header().Get("Content-Type") != "image/webp" {
		t.Fatalf("thumbnail=%d %s", thumb.Code, thumb.Body)
	}
	cfg, format, err := image.DecodeConfig(bytes.NewReader(thumb.Body.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if format != "webp" || cfg.Width > 640 || cfg.Height > 640 {
		t.Fatalf("format=%s dimensions=%dx%d", format, cfg.Width, cfg.Height)
	}
	full := get("GET", "/media/"+uploaded.StoragePath, true)
	if full.Code != 200 || !bytes.Equal(full.Body.Bytes(), original) {
		t.Fatal("original no longer served unchanged")
	}
	if thumb.Body.Len() >= len(original) {
		t.Fatalf("thumbnail %d >= original %d", thumb.Body.Len(), len(original))
	}
	t.Logf("fixture original=%d bytes, grid derivative=%d bytes", len(original), thumb.Body.Len())
	var stored string
	if err := conn.QueryRow(`SELECT thumbnail_path FROM images WHERE id=?`, uploaded.ImageID).Scan(&stored); err != nil {
		t.Fatal(err)
	}
	if stored != item.ThumbnailPath {
		t.Fatalf("persisted=%q advertised=%q", stored, item.ThumbnailPath)
	}
	if err := images.Delete(context.Background(), conn, dir, uploaded.ImageID); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, stored)); !os.IsNotExist(err) {
		t.Fatalf("thumbnail not cleaned up: %v", err)
	}
	if rr := get(http.MethodGet, url, true); rr.Code != 404 {
		t.Fatalf("deleted image status=%d", rr.Code)
	}
}
