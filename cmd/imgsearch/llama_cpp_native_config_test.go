package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLlamaNativeEmbeddingModelSpecUsesDimensions(t *testing.T) {
	spec, err := llamaNativeEmbeddingModelSpec(2048)
	if err != nil {
		t.Fatalf("native spec: %v", err)
	}
	if spec.Name != "llama.cpp-embedding" || spec.Version != "native" || spec.Dimensions != 2048 {
		t.Fatalf("unexpected native spec: %+v", spec)
	}
}

func TestLlamaNativeEmbeddingModelSpecRejectsNonPositiveDimensions(t *testing.T) {
	if _, err := llamaNativeEmbeddingModelSpec(0); err == nil {
		t.Fatal("expected non-positive dimensions error")
	}
}

func TestNewLlamaCPPNativeEmbedderRequiresModelPath(t *testing.T) {
	_, err := newLlamaCPPNativeEmbedder(llamaCPPNativeEmbedderOptions{
		VisionModelPath: "/tmp/mmproj.gguf",
		Dimensions:      2048,
		ContextSize:     8192,
		BatchSize:       512,
	})
	if err == nil {
		t.Fatal("expected missing llama-cpp-native model path error")
	}
}

func TestNewLlamaCPPNativeEmbedderRequiresVisionPath(t *testing.T) {
	tmp := t.TempDir()
	modelPath := filepath.Join(tmp, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0o644); err != nil {
		t.Fatalf("write model: %v", err)
	}

	_, err := newLlamaCPPNativeEmbedder(llamaCPPNativeEmbedderOptions{
		ModelPath:   modelPath,
		Dimensions:  2048,
		ContextSize: 8192,
		BatchSize:   512,
	})
	if err == nil {
		t.Fatal("expected missing llama-cpp-native vision path error")
	}
}

func TestNewLlamaCPPNativeEmbedderRejectsNegativeImageMaxSide(t *testing.T) {
	tmp := t.TempDir()
	modelPath := filepath.Join(tmp, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0o644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	visionPath := filepath.Join(tmp, "mmproj.gguf")
	if err := os.WriteFile(visionPath, []byte("vision"), 0o644); err != nil {
		t.Fatalf("write vision model: %v", err)
	}

	_, err := newLlamaCPPNativeEmbedder(llamaCPPNativeEmbedderOptions{
		ModelPath:       modelPath,
		VisionModelPath: visionPath,
		Dimensions:      2048,
		ContextSize:     8192,
		BatchSize:       512,
		ImageMaxSide:    -1,
	})
	if err == nil {
		t.Fatal("expected negative image max side error")
	}
}

func TestNewLlamaCPPNativeEmbedderRejectsNegativeImageMaxTokens(t *testing.T) {
	tmp := t.TempDir()
	modelPath := filepath.Join(tmp, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0o644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	visionPath := filepath.Join(tmp, "mmproj.gguf")
	if err := os.WriteFile(visionPath, []byte("vision"), 0o644); err != nil {
		t.Fatalf("write vision model: %v", err)
	}

	_, err := newLlamaCPPNativeEmbedder(llamaCPPNativeEmbedderOptions{
		ModelPath:       modelPath,
		VisionModelPath: visionPath,
		Dimensions:      2048,
		ContextSize:     8192,
		BatchSize:       512,
		ImageMaxTokens:  -1,
	})
	if err == nil {
		t.Fatal("expected negative image max tokens error")
	}
}

func TestNewLlamaCPPNativeEmbedderRejectsNegativeMaxSequences(t *testing.T) {
	tmp := t.TempDir()
	modelPath := filepath.Join(tmp, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0o644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	visionPath := filepath.Join(tmp, "mmproj.gguf")
	if err := os.WriteFile(visionPath, []byte("vision"), 0o644); err != nil {
		t.Fatalf("write vision model: %v", err)
	}

	_, err := newLlamaCPPNativeEmbedder(llamaCPPNativeEmbedderOptions{
		ModelPath:       modelPath,
		VisionModelPath: visionPath,
		Dimensions:      2048,
		ContextSize:     8192,
		BatchSize:       512,
		MaxSequences:    -1,
	})
	if err == nil {
		t.Fatal("expected negative max sequences error")
	}
}
