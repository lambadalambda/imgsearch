package app

import (
	"context"
	"database/sql"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"imgsearch/internal/httputil"
	"imgsearch/internal/vectorindex/bruteforce"
)

func TestRuntimeCollectionsAndMediaDeletion(t *testing.T) {
	conn := openDB(t)
	if err := Bootstrap(context.Background(), conn, func(context.Context, *sql.DB) error { return nil }); err != nil {
		t.Fatal(err)
	}
	data := &DataRuntime{DB: conn, Index: bruteforce.NewIndex(conn)}
	dir := t.TempDir()
	runtime, err := NewRuntime(RuntimeOptions{Data: data, DataDir: dir, ModelID: 1, Embedder: fakeRuntimeEmbedder{}, Index: data.Index})
	if err != nil {
		t.Fatal(err)
	}
	protected := httputil.NewAPIAuthMiddleware("collection-test-key")(runtime.Mux)
	call := func(method, path, body string, auth bool, want int) *httptest.ResponseRecorder {
		t.Helper()
		r := httptest.NewRequest(method, path, strings.NewReader(body))
		if auth {
			r.Header.Set(httputil.APIKeyHeaderName, "collection-test-key")
		}
		w := httptest.NewRecorder()
		protected.ServeHTTP(w, r)
		if w.Code != want {
			t.Fatalf("%s %s = %d want %d: %s", method, path, w.Code, want, w.Body)
		}
		return w
	}
	call("POST", "/api/collections", `{"name":"Saved"}`, false, 401)
	call("POST", "/api/collections", `{"name":"Saved"}`, true, 201)
	for _, q := range []string{
		`INSERT INTO images(id,sha256,original_name,storage_path,mime_type,width,height) VALUES(1,'one','one.jpg','images/one.jpg','image/jpeg',1,1)`,
		`INSERT INTO videos(id,sha256,original_name,storage_path,mime_type,width,height,duration_ms,frame_count) VALUES(1,'video','one.mp4','videos/one.mp4','video/mp4',1,1,1000,0)`,
	} {
		if _, err := conn.Exec(q); err != nil {
			t.Fatal(err)
		}
	}
	for _, path := range []string{"images/one.jpg", "videos/one.mp4"} {
		if err := os.MkdirAll(filepath.Dir(filepath.Join(dir, path)), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, path), []byte("fixture"), 0644); err != nil {
			t.Fatal(err)
		}
	}
	for _, kind := range []string{"image", "video"} {
		call("PUT", "/api/collections/1/items/"+kind+"/1", "", true, 204)
	}
	call("GET", "/api/collections/1/items", "", false, 401)
	// Deleting the list must leave both the database and files untouched.
	call("DELETE", "/api/collections/1", "", true, 204)
	for _, path := range []string{"images/one.jpg", "videos/one.mp4"} {
		if _, err := os.Stat(filepath.Join(dir, path)); err != nil {
			t.Fatal("collection deletion removed file", err)
		}
	}
	call("GET", "/api/images/1", "", true, 200)
	call("GET", "/api/videos/1", "", true, 200)
	call("POST", "/api/collections", `{"name":"Keep"}`, true, 201)
	for _, kind := range []string{"image", "video"} {
		call("PUT", "/api/collections/2/items/"+kind+"/1", "", true, 204)
	}
	for _, kind := range []string{"images", "videos"} {
		call("DELETE", fmt.Sprintf("/api/%s/1", kind), "", true, 204)
	}
	page := call("GET", "/api/collections/2/items", "", true, 200)
	if !strings.Contains(page.Body.String(), `"items":[]`) || !strings.Contains(page.Body.String(), `"total":0`) {
		t.Fatalf("dangling memberships: %s", page.Body)
	}
}
