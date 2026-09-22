package thumbnails

import (
	"context"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestThumbnailIgnoresUnrecordedCacheAfterDatabaseReset(t *testing.T) {
	_, dir, h := setup(t)
	if err := os.Mkdir(filepath.Join(dir, "images/thumbnails"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "images/thumbnails/1.webp"), []byte("old library image"), 0600); err != nil {
		t.Fatal(err)
	}
	rr := request(h, "GET", "/media/images/thumbnails/1.webp")
	if rr.Code != 200 || rr.Body.String() != "thumbnail" {
		t.Fatalf("served stale cache: %d %s", rr.Code, rr.Body)
	}
}

func TestThumbnailWaitingRequestCanCancel(t *testing.T) {
	_, _, h := setup(t)
	h.gate <- struct{}{}
	defer func() { <-h.gate }()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	req := httptest.NewRequest("GET", "/media/images/thumbnails/1.webp", nil).WithContext(ctx)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code == 200 {
		t.Fatal("served cancelled request")
	}
}

func TestThumbnailRejectsCacheDirectorySymlinkEscape(t *testing.T) {
	_, dir, h := setup(t)
	outside := t.TempDir()
	if err := os.Symlink(outside, filepath.Join(dir, "images/thumbnails")); err != nil {
		t.Fatal(err)
	}
	if rr := request(h, "GET", "/media/images/thumbnails/1.webp"); rr.Code != 404 {
		t.Fatalf("status=%d", rr.Code)
	}
	entries, err := os.ReadDir(outside)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("wrote outside data dir: %v", entries)
	}
}
