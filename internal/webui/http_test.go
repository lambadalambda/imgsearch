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
	if !strings.Contains(body, "id=\"negative-query\"") {
		t.Fatalf("expected optional negative prompt field in page")
	}
	if !strings.Contains(body, "id=\"image-lightbox\"") {
		t.Fatalf("expected lightbox container in page")
	}
	if !strings.Contains(body, "video.min.js") || !strings.Contains(body, "id=\"video-player\"") {
		t.Fatalf("expected Video.js player assets in page")
	}
	if !strings.Contains(body, "id=\"live-connection\"") {
		t.Fatalf("expected live connection indicator in page")
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
	if !strings.Contains(rr.Body.String(), "fetch('/api/videos'") {
		t.Fatalf("expected videos javascript payload")
	}
	if !strings.Contains(rr.Body.String(), "openLightbox") {
		t.Fatalf("expected lightbox javascript behavior")
	}
	if !strings.Contains(rr.Body.String(), "openVideoPlayer") {
		t.Fatalf("expected Video.js player behavior in javascript")
	}
	if !strings.Contains(rr.Body.String(), "is_anchor") {
		t.Fatalf("expected similar-search anchor behavior in javascript")
	}
	if !strings.Contains(rr.Body.String(), "params.set('neg'") {
		t.Fatalf("expected negative prompt query parameter in javascript")
	}
	if !strings.Contains(rr.Body.String(), "match_timestamp_ms") || !strings.Contains(rr.Body.String(), "preview_path") {
		t.Fatalf("expected video-result rendering fields in javascript")
	}
	if !strings.Contains(rr.Body.String(), "const mediaURL = escapeHTML(toMediaURL(item.storage_path));") {
		t.Fatalf("expected mediaURL helper in card rendering to avoid video player runtime errors")
	}
	if !strings.Contains(rr.Body.String(), "data-player-src=\"${mediaURL}\"") {
		t.Fatalf("expected player source binding to use mediaURL in card rendering")
	}
	if !strings.Contains(rr.Body.String(), "totalPages(state.imagesTotal, state.galleryPageSize)") {
		t.Fatalf("expected gallery pagination to use shared totalPages helper")
	}
	if strings.Contains(rr.Body.String(), "totalGalleryPages()") {
		t.Fatalf("expected removed totalGalleryPages helper not to be referenced in app javascript")
	}
	if !strings.Contains(rr.Body.String(), "deleteMedia(kind, id, name)") || !strings.Contains(rr.Body.String(), "fetch(`/api/${kind === 'video' ? 'videos' : 'images'}/${id}`") {
		t.Fatalf("expected delete media workflow in javascript")
	}
	if !strings.Contains(rr.Body.String(), "delete-action") {
		t.Fatalf("expected delete action class in card rendering")
	}
	if !strings.Contains(rr.Body.String(), "const supportText =") {
		t.Fatalf("expected card rendering to choose a single support text slot")
	}
	if !strings.Contains(rr.Body.String(), "supporting-text") {
		t.Fatalf("expected compact supporting text class in card rendering")
	}
	if !strings.Contains(rr.Body.String(), "tag-chip-more") {
		t.Fatalf("expected compact tag overflow chip support in card rendering")
	}
	if strings.Contains(rr.Body.String(), "description-toggle") || strings.Contains(rr.Body.String(), "Show more") {
		t.Fatalf("expected inline description toggle pattern removed from card rendering")
	}
}

func TestStylesIncludeTightRadiusAndCardDensityRules(t *testing.T) {
	h := NewHandler(t.TempDir())

	req := httptest.NewRequest(http.MethodGet, "/assets/styles.css", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusOK, rr.Body.String())
	}

	body := rr.Body.String()
	if !strings.Contains(body, "--radius-panel:") || !strings.Contains(body, "--radius-control:") {
		t.Fatalf("expected two-tier radius tokens for panel and control scales")
	}
	if strings.Contains(body, "--radius-lg:") || strings.Contains(body, "--radius-md:") || strings.Contains(body, "--radius-sm:") {
		t.Fatalf("expected legacy multi-size radius tokens to be removed")
	}
	if !strings.Contains(body, "grid-template-rows: 220px minmax(172px, 172px);") {
		t.Fatalf("expected cards to use fixed resting media/meta proportions")
	}
	if !strings.Contains(body, ".supporting-text") {
		t.Fatalf("expected dedicated supporting text clamping rules")
	}
	if !strings.Contains(body, "max-height: 1.8em;") {
		t.Fatalf("expected tag row to clamp at rest")
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
