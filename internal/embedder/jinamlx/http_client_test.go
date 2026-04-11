package jinamlx

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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
