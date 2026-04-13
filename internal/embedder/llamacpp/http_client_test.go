package llamacpp

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEmbedTextUsesV1EmbeddingsEndpoint(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method: got=%s want=POST", r.Method)
		}
		if r.URL.Path != "/v1/embeddings" {
			t.Fatalf("path: got=%s", r.URL.Path)
		}

		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		input, _ := req["input"].(string)
		if !strings.Contains(input, "query-instruction") {
			t.Fatalf("input should contain query instruction, got=%q", input)
		}
		if !strings.Contains(input, "cat on sofa") {
			t.Fatalf("input should contain query text, got=%q", input)
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []any{map[string]any{"index": 0, "embedding": []float64{0.1, 0.2, 0.3}}},
		})
	}))
	defer srv.Close()

	c, err := New(Config{
		BaseURL:          srv.URL,
		Dimensions:       3,
		QueryInstruction: "query-instruction",
	})
	if err != nil {
		t.Fatalf("new client: %v", err)
	}

	vec, err := c.EmbedText(context.Background(), "cat on sofa")
	if err != nil {
		t.Fatalf("embed text: %v", err)
	}
	if len(vec) != 3 {
		t.Fatalf("expected 3 dims, got %d", len(vec))
	}
}

func TestEmbedImageSendsMultimodalInput(t *testing.T) {
	rawImage := []byte("fake-image-bytes")
	tmp := t.TempDir()
	imagePath := filepath.Join(tmp, "sample.jpg")
	if err := os.WriteFile(imagePath, rawImage, 0o644); err != nil {
		t.Fatalf("write image: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			t.Fatalf("path: got=%s", r.URL.Path)
		}

		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}

		input, ok := req["input"].(map[string]any)
		if !ok {
			t.Fatalf("input should be an object, got=%T", req["input"])
		}
		prompt, _ := input["prompt_string"].(string)
		if !strings.Contains(prompt, "image-instruction") || !strings.Contains(prompt, "<__media__>") {
			t.Fatalf("unexpected prompt_string: %q", prompt)
		}
		multi, _ := input["multimodal_data"].([]any)
		if len(multi) != 1 {
			t.Fatalf("expected one multimodal entry, got %d", len(multi))
		}
		gotB64, _ := multi[0].(string)
		if gotB64 != base64.StdEncoding.EncodeToString(rawImage) {
			t.Fatalf("unexpected multimodal_data payload")
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode([]any{
			map[string]any{"index": 0, "embedding": []any{[]float64{1, 2, 3}}},
		})
	}))
	defer srv.Close()

	c, err := New(Config{
		BaseURL:            srv.URL,
		Dimensions:         3,
		PassageInstruction: "image-instruction",
	})
	if err != nil {
		t.Fatalf("new client: %v", err)
	}

	vec, err := c.EmbedImage(context.Background(), imagePath)
	if err != nil {
		t.Fatalf("embed image: %v", err)
	}
	if len(vec) != 3 {
		t.Fatalf("expected 3 dims, got %d", len(vec))
	}
}

func TestEmbedReturnsErrorOnDimensionMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []any{map[string]any{"index": 0, "embedding": []float64{0.1, 0.2}}},
		})
	}))
	defer srv.Close()

	c, err := New(Config{BaseURL: srv.URL, Dimensions: 3})
	if err != nil {
		t.Fatalf("new client: %v", err)
	}

	_, err = c.EmbedText(context.Background(), "hello")
	if err == nil || !strings.Contains(err.Error(), "dimension mismatch") {
		t.Fatalf("expected dimension mismatch error, got %v", err)
	}
}

func TestEmbedReturnsHTTPErrorDetails(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	defer srv.Close()

	c, err := New(Config{BaseURL: srv.URL, Dimensions: 3})
	if err != nil {
		t.Fatalf("new client: %v", err)
	}

	_, err = c.EmbedText(context.Background(), "hello")
	if err == nil || !strings.Contains(err.Error(), "status") {
		t.Fatalf("expected HTTP status error, got %v", err)
	}
}
