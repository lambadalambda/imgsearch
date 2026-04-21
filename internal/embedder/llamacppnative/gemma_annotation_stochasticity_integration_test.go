//go:build cgo

package llamacppnative

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

func TestNativeGemmaAnnotationSamplingProducesDifferentDescriptions(t *testing.T) {
	if os.Getenv("RUN_LLAMACPP_NATIVE_GEMMA_STOCHASTICITY_TEST") != "1" {
		t.Skip("set RUN_LLAMACPP_NATIVE_GEMMA_STOCHASTICITY_TEST=1 with LLAMA_NATIVE_GEMMA_MODEL_PATH and LLAMA_NATIVE_GEMMA_MMPROJ_PATH")
	}

	attempts := envIntOrDefault(t, "LLAMA_NATIVE_GEMMA_STOCHASTICITY_ATTEMPTS", 8)
	if attempts < 2 {
		t.Fatalf("LLAMA_NATIVE_GEMMA_STOCHASTICITY_ATTEMPTS must be >= 2; got %d", attempts)
	}
	temperature := envFloat32OrDefault(t, "LLAMA_NATIVE_GEMMA_STOCHASTICITY_TEMPERATURE", 1.0)
	if temperature <= 0 {
		t.Fatalf("LLAMA_NATIVE_GEMMA_STOCHASTICITY_TEMPERATURE must be > 0; got %v", temperature)
	}

	runtime := loadGemmaNativeRuntimeForSamplingTest(t, temperature, -1)
	t.Cleanup(func() { _ = runtime.Close() })

	root := findRepoRoot(t)
	imagePath := filepath.Join(root, "fixtures", "images", "woman_office.jpg")

	descriptionCounts := map[string]int{}
	for i := 0; i < attempts; i++ {
		annotation, err := runtime.AnnotateImage(context.Background(), imagePath)
		if err != nil {
			t.Fatalf("annotate attempt %d/%d: %v", i+1, attempts, err)
		}
		normalized := normalizeDescriptionVariant(annotation.Description)
		if normalized == "" {
			t.Fatalf("annotate attempt %d/%d returned empty description", i+1, attempts)
		}
		descriptionCounts[normalized]++
	}

	if len(descriptionCounts) < 2 {
		t.Fatalf("expected at least 2 unique descriptions across %d attempts at temperature=%.2f and seed=-1; got %d unique descriptions (%s)", attempts, temperature, len(descriptionCounts), summarizeDescriptionVariants(descriptionCounts))
	}
}

func loadGemmaNativeRuntimeForSamplingTest(t *testing.T, annotationTemperature float32, annotationSeed int64) *nativeGemmaRuntime {
	t.Helper()

	modelPath := strings.TrimSpace(os.Getenv("LLAMA_NATIVE_GEMMA_MODEL_PATH"))
	if modelPath == "" {
		t.Skip("set LLAMA_NATIVE_GEMMA_MODEL_PATH to the Gemma GGUF model path")
	}
	visionPath := strings.TrimSpace(os.Getenv("LLAMA_NATIVE_GEMMA_MMPROJ_PATH"))
	if visionPath == "" {
		t.Skip("set LLAMA_NATIVE_GEMMA_MMPROJ_PATH to the Gemma mmproj GGUF path")
	}

	runtime, err := newGemmaNativeRuntime(nativeGemmaRuntimeConfig{
		ModelPath:             modelPath,
		VisionModelPath:       visionPath,
		GPULayers:             envIntOrDefault(t, "LLAMA_NATIVE_GEMMA_GPU_LAYERS", 99),
		UseGPU:                envBoolOrDefault("LLAMA_NATIVE_GEMMA_USE_GPU", true),
		ContextSize:           envIntOrDefault(t, "LLAMA_NATIVE_GEMMA_CONTEXT_SIZE", 8192),
		BatchSize:             envIntOrDefault(t, "LLAMA_NATIVE_GEMMA_BATCH_SIZE", 512),
		Threads:               envIntOrDefault(t, "LLAMA_NATIVE_GEMMA_THREADS", 0),
		ImageMaxSide:          envIntOrDefault(t, "LLAMA_NATIVE_GEMMA_IMAGE_MAX_SIDE", 1024),
		ImageMaxTokens:        envIntOrDefault(t, "LLAMA_NATIVE_GEMMA_IMAGE_MAX_TOKENS", 0),
		AnnotationTemperature: annotationTemperature,
		AnnotationSeed:        annotationSeed,
	})
	if err != nil {
		t.Fatalf("new native Gemma runtime: %v", err)
	}

	return runtime
}

func envFloat32OrDefault(t *testing.T, key string, fallback float32) float32 {
	t.Helper()
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}
	parsed, err := strconv.ParseFloat(v, 32)
	if err != nil {
		t.Fatalf("parse %s as float32: %v", key, err)
	}
	return float32(parsed)
}

func normalizeDescriptionVariant(description string) string {
	trimmed := strings.TrimSpace(strings.ToLower(description))
	if trimmed == "" {
		return ""
	}
	return strings.Join(strings.Fields(trimmed), " ")
}

func summarizeDescriptionVariants(counts map[string]int) string {
	if len(counts) == 0 {
		return "none"
	}
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	if len(keys) > 3 {
		keys = keys[:3]
	}
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%q (%dx)", key, counts[key]))
	}
	return strings.Join(parts, "; ")
}
