//go:build cgo

package llamacppnative

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"testing"
)

func BenchmarkNativeEmbedImage(b *testing.B) {
	if os.Getenv("RUN_LLAMACPP_NATIVE_INTEGRATION") != "1" {
		b.Skip("set RUN_LLAMACPP_NATIVE_INTEGRATION=1 with LLAMA_NATIVE_MODEL_PATH and LLAMA_NATIVE_MMPROJ_PATH")
	}

	embedder := newNativeEmbedderForBenchmark(b)

	root := findRepoRootFromBenchmark(b)
	imagePath := filepath.Join(root, "fixtures", "images", "cat_1.jpg")

	ctx := context.Background()
	if _, err := embedder.EmbedImage(ctx, imagePath); err != nil {
		b.Fatalf("warm embed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vec, err := embedder.EmbedImage(ctx, imagePath)
		if err != nil {
			b.Fatalf("embed image: %v", err)
		}
		if len(vec) == 0 {
			b.Fatal("empty embedding")
		}
	}
}

func newNativeEmbedderForBenchmark(b *testing.B) *Embedder {
	b.Helper()

	modelPath := envOr("LLAMA_NATIVE_MODEL_PATH", "")
	if modelPath == "" {
		b.Skip("set LLAMA_NATIVE_MODEL_PATH to llama.cpp GGUF embedding model")
	}
	visionPath := envOr("LLAMA_NATIVE_MMPROJ_PATH", "")
	if visionPath == "" {
		b.Skip("set LLAMA_NATIVE_MMPROJ_PATH to llama.cpp GGUF vision projector")
	}

	e, err := New(Config{
		ModelPath:          modelPath,
		VisionModelPath:    visionPath,
		Dimensions:         envIntOrDefaultBenchmark("LLAMA_NATIVE_DIMS", 2048),
		GPULayers:          envIntOrDefaultBenchmark("LLAMA_NATIVE_GPU_LAYERS", 99),
		UseGPU:             envBoolOrDefault("LLAMA_NATIVE_USE_GPU", true),
		ContextSize:        envIntOrDefaultBenchmark("LLAMA_NATIVE_CONTEXT_SIZE", 8192),
		BatchSize:          envIntOrDefaultBenchmark("LLAMA_NATIVE_BATCH_SIZE", 512),
		Threads:            envIntOrDefaultBenchmark("LLAMA_NATIVE_THREADS", 0),
		ImageMaxSide:       envIntOrDefaultBenchmark("LLAMA_NATIVE_IMAGE_MAX_SIDE", 512),
		ImageMaxTokens:     envIntOrDefaultBenchmark("LLAMA_NATIVE_IMAGE_MAX_TOKENS", 0),
		QueryInstruction:   envOr("LLAMA_NATIVE_QUERY_INSTRUCTION", defaultQueryInstruction),
		PassageInstruction: envOr("LLAMA_NATIVE_PASSAGE_INSTRUCTION", defaultPassageInstruction),
	})
	if err != nil {
		b.Fatalf("new llama-cpp-native embedder: %v", err)
	}
	b.Cleanup(func() { _ = e.Close() })
	return e
}

func envIntOrDefaultBenchmark(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return n
}

func findRepoRootFromBenchmark(b *testing.B) string {
	b.Helper()

	_, file, _, ok := runtime.Caller(0)
	if !ok {
		b.Fatal("cannot determine caller path")
	}
	dir := filepath.Dir(file)
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			b.Fatal("could not find repository root")
		}
		dir = parent
	}
}
