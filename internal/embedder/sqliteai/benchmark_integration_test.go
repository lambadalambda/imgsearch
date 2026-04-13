package sqliteai

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func BenchmarkSQLiteAIPrepareImageForEmbedding(b *testing.B) {
	if os.Getenv("RUN_SQLITE_AI_INTEGRATION") != "1" {
		b.Skip("set RUN_SQLITE_AI_INTEGRATION=1 with SQLITE_AI_PATH, SQLITE_AI_MODEL_PATH, and SQLITE_AI_VISION_PATH")
	}
	if _, err := exec.LookPath("vips"); err != nil {
		b.Skip("vips CLI is required for sqlite-ai image preprocessing")
	}

	extensionPath := os.Getenv("SQLITE_AI_PATH")
	if extensionPath == "" {
		b.Skip("set SQLITE_AI_PATH to sqlite-ai extension binary path")
	}
	modelPath := os.Getenv("SQLITE_AI_MODEL_PATH")
	if modelPath == "" {
		b.Skip("set SQLITE_AI_MODEL_PATH to sqlite-ai GGUF embedding model")
	}
	visionPath := os.Getenv("SQLITE_AI_VISION_PATH")
	if visionPath == "" {
		b.Skip("set SQLITE_AI_VISION_PATH to sqlite-ai GGUF vision projector")
	}

	dbConn := openWithSQLiteAI(b, extensionPath)
	embedder, err := New(Config{
		DB:                 dbConn,
		ModelPath:          modelPath,
		ModelOptions:       envOr("SQLITE_AI_MODEL_OPTIONS", "gpu_layers=99"),
		VisionModelPath:    visionPath,
		VisionModelOptions: envOr("SQLITE_AI_VISION_OPTIONS", "use_gpu=1"),
		ContextOptions:     envOr("SQLITE_AI_CONTEXT_OPTIONS", defaultContextOptions),
		QueryInstruction:   envOr("SQLITE_AI_QUERY_INSTRUCTION", defaultQueryInstruction),
		PassageInstruction: envOr("SQLITE_AI_PASSAGE_INSTRUCTION", defaultPassageInstruction),
		ImageMaxSide:       envIntOrDefault("SQLITE_AI_MAX_SIDE", defaultImageMaxSide),
		VipsPath:           envOr("SQLITE_AI_VIPS_PATH", "vips"),
	})
	if err != nil {
		b.Fatalf("new sqlite-ai embedder: %v", err)
	}

	root := findRepoRoot(b)
	imagePath := filepath.Join(root, "fixtures", "images", "cat_1.jpg")

	ctx := context.Background()
	if err := embedder.ensureInitialized(ctx); err != nil {
		b.Fatalf("initialize sqlite-ai embedder: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		outPath, cleanup, err := embedder.prepareImageForEmbedding(ctx, imagePath)
		if cleanup != nil {
			cleanup()
		}
		if err != nil {
			b.Fatalf("prepare image: %v", err)
		}
		_ = outPath
	}
}

func BenchmarkSQLiteAIEmbedImage(b *testing.B) {
	if os.Getenv("RUN_SQLITE_AI_INTEGRATION") != "1" {
		b.Skip("set RUN_SQLITE_AI_INTEGRATION=1 with SQLITE_AI_PATH, SQLITE_AI_MODEL_PATH, and SQLITE_AI_VISION_PATH")
	}
	if _, err := exec.LookPath("vips"); err != nil {
		b.Skip("vips CLI is required for sqlite-ai image preprocessing")
	}

	extensionPath := os.Getenv("SQLITE_AI_PATH")
	if extensionPath == "" {
		b.Skip("set SQLITE_AI_PATH to sqlite-ai extension binary path")
	}
	modelPath := os.Getenv("SQLITE_AI_MODEL_PATH")
	if modelPath == "" {
		b.Skip("set SQLITE_AI_MODEL_PATH to sqlite-ai GGUF embedding model")
	}
	visionPath := os.Getenv("SQLITE_AI_VISION_PATH")
	if visionPath == "" {
		b.Skip("set SQLITE_AI_VISION_PATH to sqlite-ai GGUF vision projector")
	}

	dbConn := openWithSQLiteAI(b, extensionPath)
	embedder, err := New(Config{
		DB:                 dbConn,
		ModelPath:          modelPath,
		ModelOptions:       envOr("SQLITE_AI_MODEL_OPTIONS", "gpu_layers=99"),
		VisionModelPath:    visionPath,
		VisionModelOptions: envOr("SQLITE_AI_VISION_OPTIONS", "use_gpu=1"),
		ContextOptions:     envOr("SQLITE_AI_CONTEXT_OPTIONS", defaultContextOptions),
		QueryInstruction:   envOr("SQLITE_AI_QUERY_INSTRUCTION", defaultQueryInstruction),
		PassageInstruction: envOr("SQLITE_AI_PASSAGE_INSTRUCTION", defaultPassageInstruction),
		ImageMaxSide:       envIntOrDefault("SQLITE_AI_MAX_SIDE", defaultImageMaxSide),
		VipsPath:           envOr("SQLITE_AI_VIPS_PATH", "vips"),
	})
	if err != nil {
		b.Fatalf("new sqlite-ai embedder: %v", err)
	}

	root := findRepoRoot(b)
	imagePath := filepath.Join(root, "fixtures", "images", "cat_1.jpg")

	ctx := context.Background()
	// Warm model load and caches so per-iteration timing reflects steady-state.
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
