package main

import "testing"

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
	tests := []string{"jina-mlx", "jina-torch"}
	for _, kind := range tests {
		_, err := newEmbedder(kind, "", 2048, "auto")
		if err == nil {
			t.Fatalf("expected error for empty URL for %s", kind)
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
}
