package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestServerMuxServesUIAndMedia(t *testing.T) {
	dataDir := t.TempDir()
	imagesDir := filepath.Join(dataDir, "images")
	if err := os.MkdirAll(imagesDir, 0o755); err != nil {
		t.Fatalf("mkdir images dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(imagesDir, "probe.txt"), []byte("probe"), 0o644); err != nil {
		t.Fatalf("write probe file: %v", err)
	}

	mux := newServerMux(nil, dataDir, 0, nil, nil, nil)

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

	liveReq := httptest.NewRequest(http.MethodGet, "/api/live", nil)
	liveRR := httptest.NewRecorder()
	mux.ServeHTTP(liveRR, liveReq)
	if liveRR.Code != http.StatusInternalServerError {
		t.Fatalf("live status: got=%d want=%d", liveRR.Code, http.StatusInternalServerError)
	}
}

func TestServerMuxAPIRoutesRequireTokenWhenConfigured(t *testing.T) {
	dataDir := t.TempDir()
	h := withAPISecurity(newServerMux(nil, dataDir, 0, nil, nil, nil), "secret-token")

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
	if authRR.Code != http.StatusInternalServerError {
		t.Fatalf("authorized status should reach handler: got=%d want=%d body=%s", authRR.Code, http.StatusInternalServerError, authRR.Body.String())
	}
}

func TestServerMuxWebRequestsSetAuthCookieWhenConfigured(t *testing.T) {
	dataDir := t.TempDir()
	h := withAPISecurity(newServerMux(nil, dataDir, 0, nil, nil, nil), "secret-token")

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
	if apiRR.Code != http.StatusInternalServerError {
		t.Fatalf("cookie-auth status should reach handler: got=%d want=%d body=%s", apiRR.Code, http.StatusInternalServerError, apiRR.Body.String())
	}
}

func TestServerMuxDoesNotRequireTokenWhenNotConfigured(t *testing.T) {
	dataDir := t.TempDir()
	h := withAPISecurity(newServerMux(nil, dataDir, 0, nil, nil, nil), "")

	req := httptest.NewRequest(http.MethodGet, "/api/live", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("status should reach handler without auth configured: got=%d want=%d body=%s", rr.Code, http.StatusInternalServerError, rr.Body.String())
	}
}

func TestValidateHTTPExposure(t *testing.T) {
	if err := validateHTTPExposure("127.0.0.1:8080", ""); err != nil {
		t.Fatalf("loopback address should not require api key: %v", err)
	}
	if err := validateHTTPExposure("localhost:8080", ""); err != nil {
		t.Fatalf("localhost should not require api key: %v", err)
	}
	if err := validateHTTPExposure("0.0.0.0:8080", ""); err == nil {
		t.Fatalf("non-loopback address should require api key")
	}
	if err := validateHTTPExposure("0.0.0.0:8080", "secret-token"); err != nil {
		t.Fatalf("non-loopback address with api key should be allowed: %v", err)
	}
	if err := validateHTTPExposure(":8080", ""); err == nil {
		t.Fatalf("all-interface shorthand should require api key")
	}
	if err := validateHTTPExposure("[::1]:8080", ""); err != nil {
		t.Fatalf("ipv6 loopback should not require api key: %v", err)
	}
	if err := validateHTTPExposure("[::]:8080", ""); err == nil {
		t.Fatalf("ipv6 all-interface address should require api key")
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
