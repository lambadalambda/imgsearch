package webui

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRootServesAtelierShell(t *testing.T) {
	h := NewHandler(t.TempDir())

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d want=%d", rr.Code, http.StatusOK)
	}
	if ct := rr.Header().Get("Content-Type"); !strings.Contains(ct, "text/html") {
		t.Fatalf("expected html content type, got %q", ct)
	}
	body := rr.Body.String()
	// Either the built Atelier shell (post-build) or the friendly "not built"
	// placeholder is acceptable. Both must mention imgsearch and never fall
	// through to the legacy markup.
	if !strings.Contains(body, "imgsearch") {
		t.Fatalf("expected imgsearch branding in atelier shell, got body=%s", body)
	}
	if strings.Contains(body, "id=\"upload-form\"") {
		t.Fatalf("expected legacy upload form not to render at root; legacy lives at /legacy")
	}
}

func TestAtelierAssetsRouteIsRegistered(t *testing.T) {
	h := NewHandler(t.TempDir())

	// /assets/anything 404s when no Atelier build is present (only the
	// .gitkeep sentinel exists). What we're verifying is that the route is
	// wired and not falling through to the legacy asset tree.
	req := httptest.NewRequest(http.MethodGet, "/assets/should-not-exist.css", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusNotFound {
		t.Fatalf("expected 404 from atelier asset route on missing file, got %d", rr.Code)
	}
	if strings.Contains(rr.Body.String(), "loadMediaCollection") {
		t.Fatalf("atelier /assets/* must not leak legacy app.js content")
	}
}

func TestLegacyShellIsAvailableAtLegacyPath(t *testing.T) {
	h := NewHandler(t.TempDir())

	req := httptest.NewRequest(http.MethodGet, "/legacy", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d want=%d", rr.Code, http.StatusOK)
	}
	if ct := rr.Header().Get("Content-Type"); !strings.Contains(ct, "text/html") {
		t.Fatalf("expected html content type, got %q", ct)
	}
	body := rr.Body.String()
	if !strings.Contains(body, "<title>imgsearch</title>") {
		t.Fatalf("expected legacy html title, got body=%s", body)
	}
	if !strings.Contains(body, "id=\"upload-form\"") {
		t.Fatalf("expected legacy upload form to render at /legacy")
	}
	if !strings.Contains(body, "id=\"image-lightbox\"") {
		t.Fatalf("expected legacy lightbox to render at /legacy")
	}
	if !strings.Contains(body, "id=\"video-player\"") {
		t.Fatalf("expected legacy video player to render at /legacy")
	}
	if !strings.Contains(body, "/legacy/assets/styles.css") || !strings.Contains(body, "/legacy/assets/app.js") {
		t.Fatalf("expected legacy index to load assets from /legacy/assets/* so they resolve under the legacy mount")
	}
}

func TestLegacyAssetsAreServed(t *testing.T) {
	h := NewHandler(t.TempDir())

	req := httptest.NewRequest(http.MethodGet, "/legacy/assets/app.js", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusOK, rr.Body.String())
	}
	body := rr.Body.String()
	if !strings.Contains(body, "loadMediaCollection") || !strings.Contains(body, "endpoint: '/api/images'") {
		t.Fatalf("expected legacy app.js content under /legacy/assets/")
	}
	if !strings.Contains(body, "openLightbox") {
		t.Fatalf("expected legacy lightbox helper")
	}
	if !strings.Contains(body, "openVideoPlayer") {
		t.Fatalf("expected legacy video player helper")
	}
	if !strings.Contains(body, "fetch(`/api/${kind === 'video' ? 'videos' : 'images'}/${id}`") {
		t.Fatalf("expected legacy delete media workflow")
	}
}

func TestLegacyStylesAreServed(t *testing.T) {
	h := NewHandler(t.TempDir())

	req := httptest.NewRequest(http.MethodGet, "/legacy/assets/styles.css", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusOK, rr.Body.String())
	}
	body := rr.Body.String()
	if !strings.Contains(body, "--radius-panel:") || !strings.Contains(body, "--radius-control:") {
		t.Fatalf("expected legacy radius tokens in styles")
	}
	if !strings.Contains(body, ".tag-cloud-chip") || !strings.Contains(body, ".search-tag-suggestions") {
		t.Fatalf("expected legacy tag cloud + suggestion styling rules")
	}
	if !strings.Contains(body, ".thumb-actions") || !strings.Contains(body, ".thumb-match-badge") {
		t.Fatalf("expected legacy thumbnail overlay rules")
	}
}

func TestMediaServingIsRestrictedToMediaSubdirectories(t *testing.T) {
	dataDir := t.TempDir()
	imagesDir := filepath.Join(dataDir, "images")
	videosDir := filepath.Join(dataDir, "videos")
	if err := os.MkdirAll(imagesDir, 0o755); err != nil {
		t.Fatalf("mkdir images dir: %v", err)
	}
	if err := os.MkdirAll(videosDir, 0o755); err != nil {
		t.Fatalf("mkdir videos dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(imagesDir, "probe.txt"), []byte("img"), 0o644); err != nil {
		t.Fatalf("write image probe: %v", err)
	}
	if err := os.WriteFile(filepath.Join(videosDir, "clip.txt"), []byte("vid"), 0o644); err != nil {
		t.Fatalf("write video probe: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dataDir, "imgsearch.sqlite"), []byte("db"), 0o644); err != nil {
		t.Fatalf("write db probe: %v", err)
	}

	h := NewHandler(dataDir)

	allowedImageReq := httptest.NewRequest(http.MethodGet, "/media/images/probe.txt", nil)
	allowedImageRR := httptest.NewRecorder()
	h.ServeHTTP(allowedImageRR, allowedImageReq)
	if allowedImageRR.Code != http.StatusOK || allowedImageRR.Body.String() != "img" {
		t.Fatalf("image media status: got=%d body=%q", allowedImageRR.Code, allowedImageRR.Body.String())
	}

	allowedVideoReq := httptest.NewRequest(http.MethodGet, "/media/videos/clip.txt", nil)
	allowedVideoRR := httptest.NewRecorder()
	h.ServeHTTP(allowedVideoRR, allowedVideoReq)
	if allowedVideoRR.Code != http.StatusOK || allowedVideoRR.Body.String() != "vid" {
		t.Fatalf("video media status: got=%d body=%q", allowedVideoRR.Code, allowedVideoRR.Body.String())
	}

	blockedReq := httptest.NewRequest(http.MethodGet, "/media/imgsearch.sqlite", nil)
	blockedRR := httptest.NewRecorder()
	h.ServeHTTP(blockedRR, blockedReq)
	if blockedRR.Code != http.StatusNotFound {
		t.Fatalf("non-media path should be blocked: got=%d want=%d body=%s", blockedRR.Code, http.StatusNotFound, blockedRR.Body.String())
	}

	unknownSubdirReq := httptest.NewRequest(http.MethodGet, "/media/audio/song.mp3", nil)
	unknownSubdirRR := httptest.NewRecorder()
	h.ServeHTTP(unknownSubdirRR, unknownSubdirReq)
	if unknownSubdirRR.Code != http.StatusNotFound {
		t.Fatalf("unknown media subdirectory should be blocked: got=%d want=%d", unknownSubdirRR.Code, http.StatusNotFound)
	}

	traversalReq := httptest.NewRequest(http.MethodGet, "/media/images/../imgsearch.sqlite", nil)
	traversalRR := httptest.NewRecorder()
	h.ServeHTTP(traversalRR, traversalReq)
	if traversalRR.Code == http.StatusOK {
		t.Fatalf("traversal request must not succeed: got=%d body=%q", traversalRR.Code, traversalRR.Body.String())
	}
}

func TestExtensionlessStoredMP4ServesVideoContentType(t *testing.T) {
	dataDir := t.TempDir()
	videosDir := filepath.Join(dataDir, "videos")
	if err := os.MkdirAll(videosDir, 0o755); err != nil {
		t.Fatalf("mkdir videos dir: %v", err)
	}
	mp4Data := append([]byte("\x00\x00\x00\x18ftypisom\x00\x00\x00\x01isomiso2"), make([]byte, 600)...)
	if err := os.WriteFile(filepath.Join(videosDir, "hash-without-extension"), mp4Data, 0o644); err != nil {
		t.Fatalf("write mp4 probe: %v", err)
	}

	h := NewHandler(dataDir)

	req := httptest.NewRequest(http.MethodGet, "/media/videos/hash-without-extension", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusOK, rr.Body.String())
	}
	if ct := rr.Header().Get("Content-Type"); !strings.HasPrefix(ct, "video/mp4") {
		t.Fatalf("content-type: got=%q want video/mp4", ct)
	}

	rangeReq := httptest.NewRequest(http.MethodGet, "/media/videos/hash-without-extension", nil)
	rangeReq.Header.Set("Range", "bytes=0-1")
	rangeRR := httptest.NewRecorder()
	h.ServeHTTP(rangeRR, rangeReq)
	if rangeRR.Code != http.StatusPartialContent {
		t.Fatalf("range status: got=%d want=%d body=%s", rangeRR.Code, http.StatusPartialContent, rangeRR.Body.String())
	}
	if ct := rangeRR.Header().Get("Content-Type"); !strings.HasPrefix(ct, "video/mp4") {
		t.Fatalf("range content-type: got=%q want video/mp4", ct)
	}
}

func TestMediaServesFilesFromDataDir(t *testing.T) {
	dataDir := t.TempDir()
	imagesDir := filepath.Join(dataDir, "images")
	if err := os.MkdirAll(imagesDir, 0o755); err != nil {
		t.Fatalf("mkdir images dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(imagesDir, "sample.bin"), []byte("ok"), 0o644); err != nil {
		t.Fatalf("write image sample: %v", err)
	}

	h := NewHandler(dataDir)
	req := httptest.NewRequest(http.MethodGet, "/media/images/sample.bin", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusOK, rr.Body.String())
	}
	if rr.Body.String() != "ok" {
		t.Fatalf("unexpected media body: %q", rr.Body.String())
	}
}

func TestUnknownPathReturnsNotFound(t *testing.T) {
	h := NewHandler(t.TempDir())

	req := httptest.NewRequest(http.MethodGet, "/nope", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusNotFound {
		t.Fatalf("status: got=%d want=%d", rr.Code, http.StatusNotFound)
	}
}
