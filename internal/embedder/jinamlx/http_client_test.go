package jinamlx

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEmbedTextCallsServerAndReturnsVector(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method: got=%s want=POST", r.Method)
		}
		if r.URL.Path != "/embed/text" {
			t.Fatalf("path: got=%s", r.URL.Path)
		}

		var req map[string]string
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req["text"] != "cat on a sofa" {
			t.Fatalf("text: got=%q", req["text"])
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"embedding": []float32{0.1, 0.2, 0.3}})
	}))
	defer srv.Close()

	c := NewHTTPClient(srv.URL)
	vec, err := c.EmbedText(context.Background(), "cat on a sofa")
	if err != nil {
		t.Fatalf("embed text: %v", err)
	}
	if len(vec) != 3 {
		t.Fatalf("expected 3 dims, got %d", len(vec))
	}
}

func TestEmbedImageCallsServerAndReturnsVector(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embed/image" {
			t.Fatalf("path: got=%s", r.URL.Path)
		}
		var req map[string]string
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req["path"] != "/tmp/cat.jpg" {
			t.Fatalf("path payload: got=%q", req["path"])
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"embedding": []float32{1, 2}})
	}))
	defer srv.Close()

	c := NewHTTPClient(srv.URL)
	vec, err := c.EmbedImage(context.Background(), "/tmp/cat.jpg")
	if err != nil {
		t.Fatalf("embed image: %v", err)
	}
	if len(vec) != 2 {
		t.Fatalf("expected 2 dims, got %d", len(vec))
	}
}

func TestEmbedTextReturnsErrorOnNon200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "bad", http.StatusBadRequest)
	}))
	defer srv.Close()

	c := NewHTTPClient(srv.URL)
	_, err := c.EmbedText(context.Background(), "x")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "status") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEmbedImageBytesModeCallsServerAndReturnsVector(t *testing.T) {
	var sawFilename string
	var sawImageB64 string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embed/image-bytes" {
			t.Fatalf("path: got=%s", r.URL.Path)
		}
		var req map[string]string
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		sawFilename = req["filename"]
		sawImageB64 = req["image_b64"]

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"embedding": []float32{1, 2, 3}})
	}))
	defer srv.Close()

	tmp := t.TempDir()
	path := filepath.Join(tmp, "sample.jpg")
	if err := os.WriteFile(path, []byte("sample-bytes"), 0o644); err != nil {
		t.Fatalf("write sample image: %v", err)
	}

	c := NewHTTPClientWithImageMode(srv.URL, "bytes")
	vec, err := c.EmbedImage(context.Background(), path)
	if err != nil {
		t.Fatalf("embed image bytes mode: %v", err)
	}
	if len(vec) != 3 {
		t.Fatalf("expected 3 dims, got %d", len(vec))
	}
	if sawFilename != "sample.jpg" {
		t.Fatalf("filename payload: got=%q", sawFilename)
	}
	if sawImageB64 == "" {
		t.Fatal("expected non-empty image_b64 payload")
	}
}

func TestEmbedImageAutoFallsBackToBytesAndSticks(t *testing.T) {
	pathCalls := 0
	bytesCalls := 0

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/embed/image":
			pathCalls++
			http.Error(w, `{"error":"image path outside allowed directories"}`, http.StatusForbidden)
		case "/embed/image-bytes":
			bytesCalls++
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"embedding": []float32{7, 8}})
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	tmp := t.TempDir()
	path := filepath.Join(tmp, "sample.jpg")
	if err := os.WriteFile(path, []byte("sample-bytes"), 0o644); err != nil {
		t.Fatalf("write sample image: %v", err)
	}

	c := NewHTTPClientWithImageMode(srv.URL, "auto")
	if _, err := c.EmbedImage(context.Background(), path); err != nil {
		t.Fatalf("embed image first call: %v", err)
	}
	if _, err := c.EmbedImage(context.Background(), path); err != nil {
		t.Fatalf("embed image second call: %v", err)
	}

	if pathCalls != 1 {
		t.Fatalf("expected path call once before fallback, got %d", pathCalls)
	}
	if bytesCalls != 2 {
		t.Fatalf("expected bytes endpoint twice, got %d", bytesCalls)
	}
}
