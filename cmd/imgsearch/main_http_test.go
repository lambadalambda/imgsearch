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

	mediaReq := httptest.NewRequest(http.MethodGet, "/media/probe.txt", nil)
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
