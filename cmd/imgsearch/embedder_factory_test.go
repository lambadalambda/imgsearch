package main

import (
	"database/sql"
	"os"
	"path/filepath"
	"testing"

	_ "github.com/mattn/go-sqlite3"
)

func TestNewEmbedderDeterministic(t *testing.T) {
	e, err := newEmbedder("deterministic", "", 16, "auto")
	if err != nil {
		t.Fatalf("new embedder: %v", err)
	}

	v, err := e.EmbedText(t.Context(), "hello")
	if err != nil {
		t.Fatalf("embed text: %v", err)
	}
	if len(v) != 16 {
		t.Fatalf("expected dim 16, got %d", len(v))
	}
}

func TestNewEmbedderJinaRequiresURL(t *testing.T) {
	tests := []struct {
		kind       string
		dimensions int
	}{
		{kind: "jina-mlx", dimensions: 2048},
		{kind: "jina-torch", dimensions: 2048},
		{kind: "qwen3-vl-embedding-8b", dimensions: 4096},
	}
	for _, tc := range tests {
		_, err := newEmbedder(tc.kind, "", tc.dimensions, "auto")
		if err == nil {
			t.Fatalf("expected error for empty URL for %s", tc.kind)
		}
	}
}

func TestNewEmbedderRejectsUnknownType(t *testing.T) {
	_, err := newEmbedder("unknown", "http://127.0.0.1:9009", 2048, "auto")
	if err == nil {
		t.Fatal("expected unknown embedder error")
	}
}

func TestNewEmbedderJinaRejectsUnexpectedDimensions(t *testing.T) {
	tests := []string{"jina-mlx", "jina-torch"}
	for _, kind := range tests {
		_, err := newEmbedder(kind, "http://127.0.0.1:9009", 1024, "auto")
		if err == nil {
			t.Fatalf("expected dimensions mismatch error for %s", kind)
		}
	}
}

func TestNewEmbedderQwenRejectsUnexpectedDimensions(t *testing.T) {
	_, err := newEmbedder("qwen3-vl-embedding-8b", "http://127.0.0.1:9009", 2048, "auto")
	if err == nil {
		t.Fatal("expected dimensions mismatch error for qwen3-vl-embedding-8b")
	}
}

func TestNewEmbedderRejectsUnknownImageMode(t *testing.T) {
	_, err := newEmbedder("jina-torch", "http://127.0.0.1:9009", 2048, "weird")
	if err == nil {
		t.Fatal("expected unsupported image mode error")
	}
}

func TestEmbeddingModelSpecUsesEmbedderType(t *testing.T) {
	det, err := embeddingModelSpec("deterministic", 16)
	if err != nil {
		t.Fatalf("deterministic spec: %v", err)
	}
	if det.Name != "deterministic-sha256" || det.Dimensions != 16 {
		t.Fatalf("unexpected deterministic spec: %+v", det)
	}

	jina, err := embeddingModelSpec("jina-mlx", 2048)
	if err != nil {
		t.Fatalf("jina spec: %v", err)
	}
	if jina.Name != "jina-embeddings-v4" || jina.Version != "mlx-8bit" {
		t.Fatalf("unexpected jina spec: %+v", jina)
	}

	torch, err := embeddingModelSpec("jina-torch", 2048)
	if err != nil {
		t.Fatalf("jina torch spec: %v", err)
	}
	if torch.Name != "jina-embeddings-v4" || torch.Version != "torch" {
		t.Fatalf("unexpected jina torch spec: %+v", torch)
	}

	qwen, err := embeddingModelSpec("qwen3-vl-embedding-8b", 4096)
	if err != nil {
		t.Fatalf("qwen spec: %v", err)
	}
	if qwen.Name != "Qwen3-VL-Embedding-8B" || qwen.Version != "transformers" {
		t.Fatalf("unexpected qwen spec: %+v", qwen)
	}

	llama, err := embeddingModelSpec("llama-cpp", 2048)
	if err != nil {
		t.Fatalf("llama-cpp spec: %v", err)
	}
	if llama.Name != "llama.cpp-embedding" || llama.Version != "server" {
		t.Fatalf("unexpected llama-cpp spec: %+v", llama)
	}
}

func TestEmbedderDimensionsForType(t *testing.T) {
	if got, err := embedderDimensionsForType("jina-mlx"); err != nil || got != 2048 {
		t.Fatalf("jina-mlx dims: got=%d err=%v", got, err)
	}
	if got, err := embedderDimensionsForType("jina-torch"); err != nil || got != 2048 {
		t.Fatalf("jina-torch dims: got=%d err=%v", got, err)
	}
	if got, err := embedderDimensionsForType("qwen3-vl-embedding-8b"); err != nil || got != 4096 {
		t.Fatalf("qwen dims: got=%d err=%v", got, err)
	}
	if got, err := embedderDimensionsForType("deterministic"); err != nil || got != 2048 {
		t.Fatalf("deterministic dims: got=%d err=%v", got, err)
	}
	if got, err := embedderDimensionsForType("llama-cpp"); err != nil || got != 2048 {
		t.Fatalf("llama-cpp dims: got=%d err=%v", got, err)
	}
	if _, err := embedderDimensionsForType("unknown"); err == nil {
		t.Fatal("expected error for unknown embedder")
	}
}

func TestNewLlamaCPPEmbedderRequiresURL(t *testing.T) {
	_, err := newLlamaCPPEmbedder(llamaCPPEmbedderOptions{Dimensions: 2048})
	if err == nil {
		t.Fatal("expected missing llama-cpp URL error")
	}
}

func TestNewLlamaCPPEmbedderRejectsNonPositiveDimensions(t *testing.T) {
	_, err := newLlamaCPPEmbedder(llamaCPPEmbedderOptions{URL: "http://127.0.0.1:8081", Dimensions: 0})
	if err == nil {
		t.Fatal("expected non-positive llama-cpp dimensions error")
	}
}

func TestNewSQLiteAIEmbedderRequiresModelPath(t *testing.T) {
	dbConn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = dbConn.Close() })

	_, err = newSQLiteAIEmbedder(dbConn, sqliteAIEmbedderOptions{})
	if err == nil {
		t.Fatal("expected missing model path error")
	}
}

func TestNewSQLiteAIEmbedderRequiresVisionModelPath(t *testing.T) {
	dbConn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = dbConn.Close() })

	modelPath := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0o644); err != nil {
		t.Fatalf("write model: %v", err)
	}

	_, err = newSQLiteAIEmbedder(dbConn, sqliteAIEmbedderOptions{ModelPath: modelPath})
	if err == nil {
		t.Fatal("expected missing vision model path error")
	}
}

func TestNewSQLiteAIEmbedderRejectsNonPositiveImageMaxSide(t *testing.T) {
	dbConn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = dbConn.Close() })

	tmp := t.TempDir()
	modelPath := filepath.Join(tmp, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0o644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	visionPath := filepath.Join(tmp, "vision.gguf")
	if err := os.WriteFile(visionPath, []byte("vision"), 0o644); err != nil {
		t.Fatalf("write vision model: %v", err)
	}

	_, err = newSQLiteAIEmbedder(dbConn, sqliteAIEmbedderOptions{
		ModelPath:       modelPath,
		VisionModelPath: visionPath,
		ImageMaxSide:    0,
	})
	if err == nil {
		t.Fatal("expected non-positive image max side error")
	}
}
