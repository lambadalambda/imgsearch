package thumbnails

import (
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestThumbnailRevalidationDistinguishesReusedIDs(t *testing.T) {
	conn, dir, h := setup(t)
	url := "/media/images/thumbnails/1.webp"
	if rr := request(h, "GET", url); rr.Code != 200 {
		t.Fatalf("status=%d", rr.Code)
	}
	filename := filepath.Join(dir, "images/thumbnails/1.webp")
	stamp := time.Unix(1000, 1)
	if err := os.Chtimes(filename, stamp, stamp); err != nil {
		t.Fatal(err)
	}
	first := request(h, "GET", url)
	etag := first.Header().Get("ETag")
	if etag == "" {
		t.Fatal("missing thumbnail validator")
	}
	conditional := func() *httptest.ResponseRecorder {
		req := httptest.NewRequest("GET", url, nil)
		req.Header.Set("If-None-Match", etag)
		req.Header.Set("If-Modified-Since", first.Header().Get("Last-Modified"))
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, req)
		return rr
	}
	if rr := conditional(); rr.Code != 304 || rr.Body.Len() != 0 {
		t.Fatalf("unchanged=%d %s", rr.Code, rr.Body)
	}
	// Same ID and byte count, changed within the same HTTP-date second.
	if _, err := conn.Exec(`UPDATE images SET thumbnail_path=NULL WHERE id=1`); err != nil {
		t.Fatal(err)
	}
	h.generate = func(src, dst string, size int) error {
		if err := os.WriteFile(dst, []byte("different"), 0600); err != nil {
			return err
		}
		stamp := time.Unix(1000, 2)
		return os.Chtimes(dst, stamp, stamp)
	}
	if rr := conditional(); rr.Code != 200 || rr.Body.String() != "different" || rr.Header().Get("ETag") == etag {
		t.Fatalf("changed=%d %s %v", rr.Code, rr.Body, rr.Header())
	}
}
