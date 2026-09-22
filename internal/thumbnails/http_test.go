package thumbnails

import (
	"context"
	"database/sql"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"

	_ "github.com/mattn/go-sqlite3"
	"imgsearch/internal/db"
	"imgsearch/internal/httputil"
)

func setup(t *testing.T) (*sql.DB, string, *handler) {
	t.Helper()
	conn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	conn.SetMaxOpenConns(1)
	t.Cleanup(func() { conn.Close() })
	if err := db.RunMigrations(context.Background(), conn); err != nil {
		t.Fatal(err)
	}
	if _, err := conn.Exec(`INSERT INTO images(id,sha256,original_name,storage_path,mime_type,width,height) VALUES(1,'abc','photo.jpg','images/abc','image/jpeg',1280,800)`); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "images"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "images/abc"), []byte("original"), 0600); err != nil {
		t.Fatal(err)
	}
	h := NewHandler(conn, dir).(*handler)
	h.generate = func(source, destination string, maxSide int) error {
		if maxSide != 640 {
			t.Errorf("maxSide=%d", maxSide)
		}
		return os.WriteFile(destination, []byte("thumbnail"), 0600)
	}
	return conn, dir, h
}

func request(h http.Handler, method, path string) *httptest.ResponseRecorder {
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(method, path, nil))
	return rr
}

func TestLazyThumbnailPersistsCachesAndRegenerates(t *testing.T) {
	conn, dir, h := setup(t)
	calls := 0
	generate := h.generate
	h.generate = func(src, dst string, size int) error { calls++; return generate(src, dst, size) }
	url := "/media/images/thumbnails/1.webp"
	for i := 0; i < 2; i++ {
		rr := request(h, "GET", url)
		if rr.Code != 200 || rr.Body.String() != "thumbnail" || rr.Header().Get("Content-Type") != "image/webp" {
			t.Fatalf("response=%d %s %v", rr.Code, rr.Body, rr.Header())
		}
		if rr.Header().Get("Cache-Control") != "private, no-cache" {
			t.Fatalf("cache=%s", rr.Header().Get("Cache-Control"))
		}
	}
	if calls != 1 {
		t.Fatalf("generated %d times", calls)
	}
	var stored string
	if err := conn.QueryRow(`SELECT thumbnail_path FROM images WHERE id=1`).Scan(&stored); err != nil {
		t.Fatal(err)
	}
	if stored != "images/thumbnails/1.webp" {
		t.Fatalf("stored=%q", stored)
	}
	rr := request(h, "HEAD", url)
	if rr.Code != 200 || rr.Body.Len() != 0 || calls != 1 {
		t.Fatalf("HEAD=%d body=%s calls=%d", rr.Code, rr.Body, calls)
	}
	if err := os.Remove(filepath.Join(dir, stored)); err != nil {
		t.Fatal(err)
	}
	if rr := request(h, "GET", url); rr.Code != 200 || calls != 2 {
		t.Fatalf("regenerate=%d calls=%d", rr.Code, calls)
	}
	original, _ := os.ReadFile(filepath.Join(dir, "images/abc"))
	if string(original) != "original" {
		t.Fatal("changed original")
	}
}

func TestLazyThumbnailConcurrentRequestsGenerateOnce(t *testing.T) {
	_, _, h := setup(t)
	var calls atomic.Int32
	generate := h.generate
	h.generate = func(src, dst string, size int) error { calls.Add(1); return generate(src, dst, size) }
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if rr := request(h, "GET", "/media/images/thumbnails/1.webp"); rr.Code != 200 {
				t.Errorf("status=%d %s", rr.Code, rr.Body)
			}
		}()
	}
	wg.Wait()
	if calls.Load() != 1 {
		t.Fatalf("generated %d times", calls.Load())
	}
}

func TestThumbnailFailureDoesNotPublishPartialFile(t *testing.T) {
	conn, dir, h := setup(t)
	h.generate = func(src, dst string, size int) error {
		_ = os.WriteFile(dst, []byte("partial"), 0600)
		return errors.New("decode failure")
	}
	rr := request(h, "GET", "/media/images/thumbnails/1.webp")
	if rr.Code != 500 {
		t.Fatalf("status=%d", rr.Code)
	}
	entries, _ := os.ReadDir(filepath.Join(dir, "images/thumbnails"))
	if len(entries) != 0 {
		t.Fatalf("partial files: %v", entries)
	}
	var stored sql.NullString
	if err := conn.QueryRow(`SELECT thumbnail_path FROM images WHERE id=1`).Scan(&stored); err != nil {
		t.Fatal(err)
	}
	if stored.Valid {
		t.Fatal("stored failed thumbnail")
	}
}

func TestThumbnailDeletedDuringGenerationIsNotPublished(t *testing.T) {
	conn, dir, h := setup(t)
	generate := h.generate
	h.generate = func(src, dst string, size int) error {
		if _, err := conn.Exec(`DELETE FROM images WHERE id=1`); err != nil {
			t.Fatal(err)
		}
		return generate(src, dst, size)
	}
	if rr := request(h, "GET", "/media/images/thumbnails/1.webp"); rr.Code != 404 {
		t.Fatalf("status=%d %s", rr.Code, rr.Body)
	}
	if _, err := os.Stat(filepath.Join(dir, "images/thumbnails/1.webp")); !os.IsNotExist(err) {
		t.Fatalf("orphan thumbnail: %v", err)
	}
}

func TestThumbnailRejectsInvalidRequestsAndEscapingPaths(t *testing.T) {
	conn, dir, h := setup(t)
	for _, url := range []string{"/media/images/thumbnails/", "/media/images/thumbnails/0.webp", "/media/images/thumbnails/2.webp", "/media/images/thumbnails/1.jpg", "/media/images/thumbnails/../abc", "/media/images/thumbnails/01.webp"} {
		if rr := request(h, "GET", url); rr.Code != 404 {
			t.Errorf("%s status=%d", url, rr.Code)
		}
	}
	if rr := request(h, "POST", "/media/images/thumbnails/1.webp"); rr.Code != 405 || rr.Header().Get("Allow") != "GET, HEAD" {
		t.Fatalf("method=%d %v", rr.Code, rr.Header())
	}
	outside := t.TempDir()
	secret := filepath.Join(outside, "secret")
	if err := os.WriteFile(secret, []byte("secret"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(secret, filepath.Join(dir, "images/link")); err != nil {
		t.Fatal(err)
	}
	for _, stored := range []string{"../secret", secret, "images/link"} {
		if _, err := conn.Exec(`UPDATE images SET storage_path=? WHERE id=1`, stored); err != nil {
			t.Fatal(err)
		}
		if rr := request(h, "GET", "/media/images/thumbnails/1.webp"); rr.Code != 404 {
			t.Errorf("%s status=%d", stored, rr.Code)
		}
	}
}

func TestThumbnailRequiresMediaAuthentication(t *testing.T) {
	_, _, h := setup(t)
	protected := httputil.NewAPIAuthMiddleware("secret-token")(h)
	rr := request(protected, "GET", "/media/images/thumbnails/1.webp")
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("anonymous status=%d", rr.Code)
	}
}
