package webui

import (
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"testing/fstest"

	"imgsearch/internal/httputil"
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
		t.Fatalf("expected no legacy upload form at root")
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

func TestBuiltAtelierEmittedAssets(t *testing.T) {
	assets, err := fs.Sub(atelierFiles, "atelier/dist")
	if err != nil {
		t.Fatal(err)
	}
	index, built := readAtelierIndex(assets)
	if !built {
		t.Skip("source-only checkout; build:frontend and CI exercise the embedded build")
	}
	urls := regexp.MustCompile(`(?:src|href)="(/assets/[^" ]+)"`).FindAllSubmatch(index, -1)
	if len(urls) < 2 {
		t.Fatalf("expected emitted JS and CSS in index: %s", index)
	}
	h := NewHandler(t.TempDir())
	for _, match := range urls {
		url := string(match[1])
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, url, nil))
		if rr.Code != http.StatusOK || rr.Body.Len() == 0 {
			t.Fatalf("asset %s: status=%d bytes=%d", url, rr.Code, rr.Body.Len())
		}
		ct := rr.Header().Get("Content-Type")
		if strings.HasSuffix(url, ".js") && !strings.Contains(ct, "javascript") || strings.HasSuffix(url, ".css") && !strings.Contains(ct, "text/css") {
			t.Fatalf("asset %s: content-type=%q", url, ct)
		}
	}
	// The favicon endpoint must not fall through to the HTML shell, whether or
	// not the current frontend supplies an icon.
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/favicon.ico", nil))
	_, err = fs.Stat(assets, "favicon.ico")
	want := http.StatusOK
	if err != nil {
		want = http.StatusNotFound
	}
	if rr.Code != want {
		t.Fatalf("favicon: got=%d want=%d", rr.Code, want)
	}
}

func TestRetiredLegacyPathsReturnNotFound(t *testing.T) {
	h := NewHandler(t.TempDir())
	for _, path := range []string{"/legacy", "/legacy/", "/legacy/index.html", "/legacy/anything", "/legacy/assets/app.js", "/legacy/assets/styles.css", "/assets/app.js", "/assets/styles.css"} {
		for _, method := range []string{http.MethodGet, http.MethodHead, http.MethodPost} {
			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, httptest.NewRequest(method, path, nil))
			if rr.Code != http.StatusNotFound {
				t.Errorf("%s %s: got=%d want=404", method, path, rr.Code)
			}
		}
	}
}

func TestMissingAtelierExplainsBuildWithoutLegacyLink(t *testing.T) {
	index, built := readAtelierIndex(fstest.MapFS{})
	if built || !strings.Contains(string(index), "mise run build:frontend") || strings.Contains(string(index), "/legacy") {
		t.Fatalf("unexpected developer fallback: built=%v body=%s", built, index)
	}
}

// Exercise the real shell/media handler with the production middleware order.
func TestAtelierRootBootstrapsAuthenticatedMedia(t *testing.T) {
	dataDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dataDir, "videos"), 0o755); err != nil {
		t.Fatal(err)
	}
	data := append([]byte("\x00\x00\x00\x18ftypisom\x00\x00\x00\x01isomiso2"), make([]byte, 600)...)
	if err := os.WriteFile(filepath.Join(dataDir, "videos", "probe"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.Handle("/", NewHandler(dataDir))
	mux.HandleFunc("/api/probe", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusNoContent) })
	h := httputil.NewAPIKeyCookieMiddleware("secret")(httputil.NewAPIAuthMiddleware("secret")(mux))
	root := httptest.NewRecorder()
	h.ServeHTTP(root, httptest.NewRequest(http.MethodGet, "/", nil))
	cookies := root.Result().Cookies()
	if root.Code != http.StatusOK || len(cookies) != 1 || !cookies[0].HttpOnly || cookies[0].Path != "/" {
		t.Fatalf("root bootstrap failed: status=%d cookies=%v", root.Code, cookies)
	}
	for _, path := range []string{"/api/probe", "/media/videos/probe"} {
		anon := httptest.NewRecorder()
		h.ServeHTTP(anon, httptest.NewRequest(http.MethodGet, path, nil))
		if anon.Code != http.StatusUnauthorized || anon.Header().Get("Set-Cookie") != "" {
			t.Fatalf("anonymous %s: %d", path, anon.Code)
		}
		req := httptest.NewRequest(http.MethodGet, path, nil)
		req.AddCookie(cookies[0])
		req.Header.Set("Range", "bytes=0-1")
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, req)
		if path == "/api/probe" {
			if rr.Code != http.StatusNoContent {
				t.Fatalf("cookie API status: %d", rr.Code)
			}
		} else if rr.Code != http.StatusPartialContent || !strings.HasPrefix(rr.Header().Get("Content-Type"), "video/mp4") || rr.Body.Len() != 2 {
			t.Fatalf("cookie media range: status=%d headers=%v body=%q", rr.Code, rr.Header(), rr.Body.String())
		}
	}
	retired := httptest.NewRecorder()
	h.ServeHTTP(retired, httptest.NewRequest(http.MethodGet, "/legacy", nil))
	if retired.Code != http.StatusNotFound {
		t.Fatalf("retired path with auth chain: %d", retired.Code)
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

func TestMediaDirectoryPathsAreNotListed(t *testing.T) {
	dataDir := t.TempDir()
	for _, sub := range []string{"images", "videos", filepath.Join("images", "nested")} {
		if err := os.MkdirAll(filepath.Join(dataDir, sub), 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", sub, err)
		}
	}
	if err := os.WriteFile(filepath.Join(dataDir, "images", "visible.bin"), []byte("img"), 0o644); err != nil {
		t.Fatalf("write image probe: %v", err)
	}
	// http.FileServer's classic listing side doors: an index.html redirect
	// and a directory reached through a symlink.
	if err := os.WriteFile(filepath.Join(dataDir, "images", "index.html"), []byte("<a href=\"visible.bin\">"), 0o644); err != nil {
		t.Fatalf("write index probe: %v", err)
	}
	if err := os.Symlink(filepath.Join(dataDir, "images", "nested"), filepath.Join(dataDir, "images", "linkdir")); err != nil {
		t.Fatalf("symlink dir: %v", err)
	}

	h := NewHandler(dataDir)

	for _, path := range []string{"/media/images/", "/media/videos/", "/media/images", "/media/videos", "/media/images/nested/", "/media/images/nested", "/media/images/linkdir/", "/media/images/linkdir", "/media/", "/media"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, req)
		if rr.Code != http.StatusNotFound {
			t.Fatalf("%s: got=%d want=%d body=%q", path, rr.Code, http.StatusNotFound, rr.Body.String())
		}
		if strings.Contains(rr.Body.String(), "visible.bin") {
			t.Fatalf("%s: response leaked directory listing: %q", path, rr.Body.String())
		}
	}

	indexReq := httptest.NewRequest(http.MethodGet, "/media/images/index.html", nil)
	indexRR := httptest.NewRecorder()
	h.ServeHTTP(indexRR, indexReq)
	if indexRR.Code == http.StatusOK && strings.Contains(indexRR.Body.String(), "visible.bin") {
		t.Fatalf("index.html path must not list: got=%d body=%q", indexRR.Code, indexRR.Body.String())
	}

	req := httptest.NewRequest(http.MethodGet, "/media/images/visible.bin", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK || rr.Body.String() != "img" {
		t.Fatalf("file still served: got=%d body=%q", rr.Code, rr.Body.String())
	}
}
