package webui

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRootServesIndexPage(t *testing.T) {
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
	if !strings.Contains(body, "<title>imgsearch</title>") {
		t.Fatalf("expected html title, got body=%s", body)
	}
	if !strings.Contains(body, "id=\"upload-form\"") {
		t.Fatalf("expected upload form in page")
	}
}

func TestAssetsAreServed(t *testing.T) {
	h := NewHandler(t.TempDir())

	req := httptest.NewRequest(http.MethodGet, "/assets/app.js", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusOK, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), "fetch('/api/images'") {
		t.Fatalf("expected app javascript payload")
	}
}

func TestMediaServesFilesFromDataImagesDir(t *testing.T) {
	dataDir := t.TempDir()
	imagesDir := filepath.Join(dataDir, "images")
	if err := os.MkdirAll(imagesDir, 0o755); err != nil {
		t.Fatalf("mkdir images dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(imagesDir, "sample.bin"), []byte("ok"), 0o644); err != nil {
		t.Fatalf("write image sample: %v", err)
	}

	h := NewHandler(dataDir)
	req := httptest.NewRequest(http.MethodGet, "/media/sample.bin", nil)
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
