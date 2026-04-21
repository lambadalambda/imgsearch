package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNewLlamaCPPNativeAnnotatorRejectsNegativeAnnotationTemperature(t *testing.T) {
	tmp := t.TempDir()
	modelPath := filepath.Join(tmp, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0o644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	visionPath := filepath.Join(tmp, "mmproj.gguf")
	if err := os.WriteFile(visionPath, []byte("vision"), 0o644); err != nil {
		t.Fatalf("write vision model: %v", err)
	}

	_, err := newLlamaCPPNativeAnnotator(llamaCPPNativeAnnotatorOptions{
		ModelPath:             modelPath,
		VisionModelPath:       visionPath,
		ContextSize:           8192,
		BatchSize:             512,
		AnnotationTemperature: -0.1,
		AnnotationSeed:        defaultLlamaNativeAnnotationSeed,
	})
	if err == nil {
		t.Fatal("expected negative annotation temperature error")
	}
}

func TestNewLlamaCPPNativeAnnotatorRejectsAnnotationSeedOutOfRange(t *testing.T) {
	tmp := t.TempDir()
	modelPath := filepath.Join(tmp, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0o644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	visionPath := filepath.Join(tmp, "mmproj.gguf")
	if err := os.WriteFile(visionPath, []byte("vision"), 0o644); err != nil {
		t.Fatalf("write vision model: %v", err)
	}

	_, err := newLlamaCPPNativeAnnotator(llamaCPPNativeAnnotatorOptions{
		ModelPath:             modelPath,
		VisionModelPath:       visionPath,
		ContextSize:           8192,
		BatchSize:             512,
		AnnotationTemperature: defaultLlamaNativeAnnotationTemperature,
		AnnotationSeed:        defaultLlamaNativeMaxAnnotationSeed + 1,
	})
	if err == nil {
		t.Fatal("expected out-of-range annotation seed error")
	}
}
