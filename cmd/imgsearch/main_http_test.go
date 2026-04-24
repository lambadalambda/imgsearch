package main

import (
	"context"
	"database/sql"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/app"
	"imgsearch/internal/vectorindex/bruteforce"
)

type fakeHTTPEmbedder struct{}

func (fakeHTTPEmbedder) EmbedText(context.Context, string) ([]float32, error) {
	return []float32{1, 0}, nil
}
func (fakeHTTPEmbedder) EmbedImage(context.Context, string) ([]float32, error) {
	return []float32{1, 0}, nil
}

func newTestRuntimeMux(t *testing.T, dataDir string) http.Handler {
	t.Helper()
	sqlDB, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	dataRuntime := &app.DataRuntime{DB: sqlDB, DBPath: ":memory:", Index: bruteforce.NewIndex(sqlDB), VectorBackend: "bruteforce"}
	if err := app.Bootstrap(context.Background(), sqlDB, func(context.Context, *sql.DB) error { return nil }); err != nil {
		_ = dataRuntime.Close()
		t.Fatalf("bootstrap: %v", err)
	}
	runtime, err := app.NewRuntime(app.RuntimeOptions{
		Data:                 dataRuntime,
		DataDir:              dataDir,
		ModelID:              1,
		Embedder:             fakeHTTPEmbedder{},
		Index:                dataRuntime.Index,
		WorkerLeaseDuration:  30 * time.Second,
		WorkerRetryBaseDelay: 5 * time.Second,
		LiveInterval:         2 * time.Second,
		LiveImagesLimit:      120,
	})
	if err != nil {
		_ = dataRuntime.Close()
		t.Fatalf("new runtime: %v", err)
	}
	t.Cleanup(func() { _ = runtime.Close() })
	return runtime.Mux
}

func newUnavailableAPIHandler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/live", func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "unavailable", http.StatusServiceUnavailable)
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	return mux
}

func TestServerMuxServesUIAndMedia(t *testing.T) {
	dataDir := t.TempDir()
	imagesDir := filepath.Join(dataDir, "images")
	if err := os.MkdirAll(imagesDir, 0o755); err != nil {
		t.Fatalf("mkdir images dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(imagesDir, "probe.txt"), []byte("probe"), 0o644); err != nil {
		t.Fatalf("write probe file: %v", err)
	}

	mux := newTestRuntimeMux(t, dataDir)

	rootReq := httptest.NewRequest(http.MethodGet, "/", nil)
	rootRR := httptest.NewRecorder()
	mux.ServeHTTP(rootRR, rootReq)
	if rootRR.Code != http.StatusOK {
		t.Fatalf("root status: got=%d want=%d", rootRR.Code, http.StatusOK)
	}
	if !strings.Contains(rootRR.Body.String(), "<title>imgsearch</title>") {
		t.Fatalf("expected title in root html")
	}

	mediaReq := httptest.NewRequest(http.MethodGet, "/media/images/probe.txt", nil)
	mediaRR := httptest.NewRecorder()
	mux.ServeHTTP(mediaRR, mediaReq)
	if mediaRR.Code != http.StatusOK {
		t.Fatalf("media status: got=%d want=%d body=%s", mediaRR.Code, http.StatusOK, mediaRR.Body.String())
	}
	if mediaRR.Body.String() != "probe" {
		t.Fatalf("unexpected media body: %q", mediaRR.Body.String())
	}

	healthReq := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	healthRR := httptest.NewRecorder()
	mux.ServeHTTP(healthRR, healthReq)
	if healthRR.Code != http.StatusOK {
		t.Fatalf("health status: got=%d want=%d", healthRR.Code, http.StatusOK)
	}
	if strings.TrimSpace(healthRR.Body.String()) != "ok" {
		t.Fatalf("unexpected health body: %q", healthRR.Body.String())
	}

}

func TestServerMuxAPIRoutesRequireTokenWhenConfigured(t *testing.T) {
	h := withAPISecurity(newUnavailableAPIHandler(), "secret-token")

	unauthReq := httptest.NewRequest(http.MethodGet, "/api/live", nil)
	unauthRR := httptest.NewRecorder()
	h.ServeHTTP(unauthRR, unauthReq)
	if unauthRR.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized status: got=%d want=%d body=%s", unauthRR.Code, http.StatusUnauthorized, unauthRR.Body.String())
	}

	unauthWSReq := httptest.NewRequest(http.MethodGet, "/api/live", nil)
	unauthWSReq.Header.Set("Connection", "Upgrade")
	unauthWSReq.Header.Set("Upgrade", "websocket")
	unauthWSReq.Header.Set("Sec-WebSocket-Version", "13")
	unauthWSReq.Header.Set("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ==")
	unauthWSRR := httptest.NewRecorder()
	h.ServeHTTP(unauthWSRR, unauthWSReq)
	if unauthWSRR.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized websocket status: got=%d want=%d body=%s", unauthWSRR.Code, http.StatusUnauthorized, unauthWSRR.Body.String())
	}

	authReq := httptest.NewRequest(http.MethodGet, "/api/live", nil)
	authReq.Header.Set("X-Imgsearch-API-Key", "secret-token")
	authRR := httptest.NewRecorder()
	h.ServeHTTP(authRR, authReq)
	if authRR.Code != http.StatusServiceUnavailable {
		t.Fatalf("authorized status should reach handler: got=%d want=%d body=%s", authRR.Code, http.StatusServiceUnavailable, authRR.Body.String())
	}
}

func TestServerMuxWebRequestsSetAuthCookieWhenConfigured(t *testing.T) {
	h := withAPISecurity(newUnavailableAPIHandler(), "secret-token")

	rootReq := httptest.NewRequest(http.MethodGet, "/", nil)
	rootRR := httptest.NewRecorder()
	h.ServeHTTP(rootRR, rootReq)
	if rootRR.Code != http.StatusOK {
		t.Fatalf("root status: got=%d want=%d", rootRR.Code, http.StatusOK)
	}
	setCookie := rootRR.Header().Get("Set-Cookie")
	if !strings.Contains(setCookie, "imgsearch_api_key=") {
		t.Fatalf("expected auth cookie, got %q", setCookie)
	}

	apiReq := httptest.NewRequest(http.MethodGet, "/api/live", nil)
	for _, c := range rootRR.Result().Cookies() {
		apiReq.AddCookie(c)
	}
	apiRR := httptest.NewRecorder()
	h.ServeHTTP(apiRR, apiReq)
	if apiRR.Code != http.StatusServiceUnavailable {
		t.Fatalf("cookie-auth status should reach handler: got=%d want=%d body=%s", apiRR.Code, http.StatusServiceUnavailable, apiRR.Body.String())
	}
}

func TestServerMuxDoesNotRequireTokenWhenNotConfigured(t *testing.T) {
	h := withAPISecurity(newUnavailableAPIHandler(), "")

	req := httptest.NewRequest(http.MethodGet, "/api/live", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("status should reach handler without auth configured: got=%d want=%d body=%s", rr.Code, http.StatusServiceUnavailable, rr.Body.String())
	}
}

func TestServerMuxDefaultAPIKeyAuthenticatesWhenConfigured(t *testing.T) {
	h := withAPISecurity(newUnavailableAPIHandler(), defaultAPIKey)

	unauthReq := httptest.NewRequest(http.MethodGet, "/api/live", nil)
	unauthRR := httptest.NewRecorder()
	h.ServeHTTP(unauthRR, unauthReq)
	if unauthRR.Code != http.StatusUnauthorized {
		t.Fatalf("expected unauthorized without header: got=%d", unauthRR.Code)
	}

	authReq := httptest.NewRequest(http.MethodGet, "/api/live", nil)
	authReq.Header.Set("X-Imgsearch-API-Key", defaultAPIKey)
	authRR := httptest.NewRecorder()
	h.ServeHTTP(authRR, authReq)
	if authRR.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected handler reached with default key: got=%d want=%d", authRR.Code, http.StatusServiceUnavailable)
	}
}

func TestValidateHTTPExposure(t *testing.T) {
	if err := validateHTTPExposure("127.0.0.1:8080", "", true); err != nil {
		t.Fatalf("loopback address should not require api key: %v", err)
	}
	if err := validateHTTPExposure("localhost:8080", "", false); err != nil {
		t.Fatalf("localhost should not require api key: %v", err)
	}
	if err := validateHTTPExposure("0.0.0.0:8080", "", false); err == nil {
		t.Fatalf("non-loopback address should require api key")
	}
	if err := validateHTTPExposure("0.0.0.0:8080", "secret-token", false); err != nil {
		t.Fatalf("non-loopback address with api key should be allowed: %v", err)
	}
	if err := validateHTTPExposure(":8080", "", false); err == nil {
		t.Fatalf("all-interface shorthand should require api key")
	}
	if err := validateHTTPExposure("[::1]:8080", "", false); err != nil {
		t.Fatalf("ipv6 loopback should not require api key: %v", err)
	}
	if err := validateHTTPExposure("[::]:8080", "", false); err == nil {
		t.Fatalf("ipv6 all-interface address should require api key")
	}
	if err := validateHTTPExposure("0.0.0.0:8080", defaultAPIKey, true); err == nil {
		t.Fatalf("non-loopback address should reject built-in development api key")
	}
}

func TestConfiguredHTTPServerUsesSafeTimeoutDefaults(t *testing.T) {
	h := http.NewServeMux()
	server := configuredHTTPServer("127.0.0.1:8080", h)

	if server.Addr != "127.0.0.1:8080" {
		t.Fatalf("addr: got=%q", server.Addr)
	}
	if server.Handler != h {
		t.Fatalf("expected configured handler to be installed")
	}
	if server.ReadHeaderTimeout != defaultHTTPReadHeaderTimeout {
		t.Fatalf("read header timeout: got=%s want=%s", server.ReadHeaderTimeout, defaultHTTPReadHeaderTimeout)
	}
	if server.ReadTimeout != defaultHTTPReadTimeout {
		t.Fatalf("read timeout: got=%s want=%s", server.ReadTimeout, defaultHTTPReadTimeout)
	}
	if server.WriteTimeout != defaultHTTPWriteTimeout {
		t.Fatalf("write timeout: got=%s want=%s", server.WriteTimeout, defaultHTTPWriteTimeout)
	}
	if server.IdleTimeout != defaultHTTPIdleTimeout {
		t.Fatalf("idle timeout: got=%s want=%s", server.IdleTimeout, defaultHTTPIdleTimeout)
	}
	if server.MaxHeaderBytes != defaultHTTPMaxHeaderBytes {
		t.Fatalf("max header bytes: got=%d want=%d", server.MaxHeaderBytes, defaultHTTPMaxHeaderBytes)
	}
}

func TestServeHTTPWithShutdownReturnsAfterContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	server := configuredHTTPServer("127.0.0.1:0", http.NewServeMux())

	done := make(chan error, 1)
	go func() {
		done <- serveHTTPWithShutdown(ctx, server)
	}()

	time.Sleep(10 * time.Millisecond)
	cancel()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("serve with shutdown: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("server shutdown did not complete after context cancellation")
	}
}

func TestServeHTTPWithShutdownReturnsListenError(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer func() { _ = listener.Close() }()

	server := configuredHTTPServer(listener.Addr().String(), http.NewServeMux())
	err = serveHTTPWithShutdown(context.Background(), server)
	if err == nil {
		t.Fatalf("expected listen error for already-bound address")
	}
}

func TestHTTPServerTimeoutConstantsAreSane(t *testing.T) {
	if defaultHTTPReadHeaderTimeout <= 0 {
		t.Fatalf("read header timeout must be positive: got=%s", defaultHTTPReadHeaderTimeout)
	}
	if defaultHTTPReadTimeout <= 0 {
		t.Fatalf("read timeout must be positive: got=%s", defaultHTTPReadTimeout)
	}
	if defaultHTTPReadTimeout < defaultHTTPReadHeaderTimeout {
		t.Fatalf("read timeout must be >= read header timeout: read=%s readHeader=%s", defaultHTTPReadTimeout, defaultHTTPReadHeaderTimeout)
	}
	if defaultHTTPWriteTimeout <= 0 {
		t.Fatalf("write timeout must be positive: got=%s", defaultHTTPWriteTimeout)
	}
	if defaultHTTPIdleTimeout <= 0 {
		t.Fatalf("idle timeout must be positive: got=%s", defaultHTTPIdleTimeout)
	}
	if defaultHTTPMaxHeaderBytes <= 0 {
		t.Fatalf("max header bytes must be positive: got=%d", defaultHTTPMaxHeaderBytes)
	}
}

func TestResolveAPIKeyUsesDefaultWhenUnset(t *testing.T) {
	apiKey, usingDefault := resolveAPIKey("   ")
	if apiKey != defaultAPIKey {
		t.Fatalf("api key: got=%q want=%q", apiKey, defaultAPIKey)
	}
	if !usingDefault {
		t.Fatalf("expected usingDefault=true when api key unset")
	}
}

func TestResolveAPIKeyKeepsExplicitValue(t *testing.T) {
	apiKey, usingDefault := resolveAPIKey("secret-token")
	if apiKey != "secret-token" {
		t.Fatalf("api key: got=%q want=%q", apiKey, "secret-token")
	}
	if usingDefault {
		t.Fatalf("expected usingDefault=false for explicit api key")
	}
}
