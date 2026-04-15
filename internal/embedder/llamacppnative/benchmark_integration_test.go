//go:build cgo

package llamacppnative

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
)

func BenchmarkNativeEmbedImage(b *testing.B) {
	if os.Getenv("RUN_LLAMACPP_NATIVE_INTEGRATION") != "1" {
		b.Skip("set RUN_LLAMACPP_NATIVE_INTEGRATION=1 with LLAMA_NATIVE_MODEL_PATH and LLAMA_NATIVE_MMPROJ_PATH")
	}

	embedder := newNativeEmbedderForBenchmark(b)

	root := findRepoRootFromBenchmark(b)
	imagePaths := benchmarkImagePaths(b, root)

	ctx := context.Background()
	if _, err := embedder.EmbedImage(ctx, imagePaths[0]); err != nil {
		b.Fatalf("warm embed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vec, err := embedder.EmbedImage(ctx, imagePaths[i%len(imagePaths)])
		if err != nil {
			b.Fatalf("embed image: %v", err)
		}
		if len(vec) == 0 {
			b.Fatal("empty embedding")
		}
	}
}

func benchmarkImagePaths(b *testing.B, repoRoot string) []string {
	b.Helper()
	if explicit := strings.TrimSpace(os.Getenv("LLAMA_NATIVE_BENCH_IMAGE_PATH")); explicit != "" {
		return []string{explicit}
	}
	if dir := strings.TrimSpace(os.Getenv("LLAMA_NATIVE_BENCH_IMAGE_DIR")); dir != "" {
		paths, err := benchmarkImagePathsFromDir(dir, envIntOrDefaultBenchmark("LLAMA_NATIVE_BENCH_IMAGE_LIMIT", 32))
		if err != nil {
			b.Fatalf("load benchmark image dir: %v", err)
		}
		if len(paths) == 0 {
			b.Fatalf("no benchmark images found in %s", dir)
		}
		return paths
	}
	return []string{filepath.Join(repoRoot, "fixtures", "images", "cat_1.jpg")}
}

func benchmarkImagePathsFromDir(dir string, limit int) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	paths := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(entry.Name()))
		switch ext {
		case ".jpg", ".jpeg", ".png", ".webp", ".avif":
			paths = append(paths, filepath.Join(dir, entry.Name()))
		}
	}
	sort.Strings(paths)
	if limit > 0 && len(paths) > limit {
		paths = paths[:limit]
	}
	return paths, nil
}

func BenchmarkNativeAnnotateImage(b *testing.B) {
	if os.Getenv("RUN_LLAMACPP_NATIVE_GEMMA_BENCH") != "1" {
		b.Skip("set RUN_LLAMACPP_NATIVE_GEMMA_BENCH=1 with LLAMA_NATIVE_GEMMA_MODEL_PATH and LLAMA_NATIVE_GEMMA_MMPROJ_PATH")
	}

	runtime := newNativeAnnotatorForBenchmark(b)

	root := findRepoRootFromBenchmark(b)
	imagePath := filepath.Join(root, "fixtures", "images", "woman_office.jpg")

	ctx := context.Background()
	annotation, err := runtime.AnnotateImage(ctx, imagePath)
	if err != nil {
		b.Fatalf("warm annotate: %v", err)
	}
	if annotation.Description == "" || len(annotation.Tags) == 0 {
		b.Fatalf("warm annotate returned incomplete result: %+v", annotation)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		annotation, err := runtime.AnnotateImage(ctx, imagePath)
		if err != nil {
			b.Fatalf("annotate image: %v", err)
		}
		if annotation.Description == "" || len(annotation.Tags) == 0 {
			b.Fatalf("empty annotation result: %+v", annotation)
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
		ContextSize:        envIntOrDefaultBenchmark("LLAMA_NATIVE_CONTEXT_SIZE", 512),
		BatchSize:          envIntOrDefaultBenchmark("LLAMA_NATIVE_BATCH_SIZE", 512),
		Threads:            envIntOrDefaultBenchmark("LLAMA_NATIVE_THREADS", 0),
		ImageMaxSide:       envIntOrDefaultBenchmark("LLAMA_NATIVE_IMAGE_MAX_SIDE", 384),
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

func newNativeAnnotatorForBenchmark(b *testing.B) *nativeGemmaRuntime {
	b.Helper()

	modelPath := envOr("LLAMA_NATIVE_GEMMA_MODEL_PATH", "")
	if modelPath == "" {
		b.Skip("set LLAMA_NATIVE_GEMMA_MODEL_PATH to the Gemma GGUF model path")
	}
	visionPath := envOr("LLAMA_NATIVE_GEMMA_MMPROJ_PATH", "")
	if visionPath == "" {
		b.Skip("set LLAMA_NATIVE_GEMMA_MMPROJ_PATH to the Gemma mmproj GGUF path")
	}

	runtime, err := newGemmaNativeRuntime(nativeGemmaRuntimeConfig{
		ModelPath:       modelPath,
		VisionModelPath: visionPath,
		GPULayers:       envIntOrDefaultBenchmark("LLAMA_NATIVE_GEMMA_GPU_LAYERS", 99),
		UseGPU:          envBoolOrDefault("LLAMA_NATIVE_GEMMA_USE_GPU", true),
		ContextSize:     envIntOrDefaultBenchmark("LLAMA_NATIVE_GEMMA_CONTEXT_SIZE", 8192),
		BatchSize:       envIntOrDefaultBenchmark("LLAMA_NATIVE_GEMMA_BATCH_SIZE", 512),
		Threads:         envIntOrDefaultBenchmark("LLAMA_NATIVE_GEMMA_THREADS", 0),
		ImageMaxSide:    envIntOrDefaultBenchmark("LLAMA_NATIVE_GEMMA_IMAGE_MAX_SIDE", 1024),
		ImageMaxTokens:  envIntOrDefaultBenchmark("LLAMA_NATIVE_GEMMA_IMAGE_MAX_TOKENS", 0),
	})
	if err != nil {
		b.Fatalf("new native Gemma runtime: %v", err)
	}
	b.Cleanup(func() { _ = runtime.Close() })
	return runtime
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
