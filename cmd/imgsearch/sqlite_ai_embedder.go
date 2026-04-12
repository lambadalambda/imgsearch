package main

import (
	"database/sql"
	"fmt"
	"os"
	"strings"

	"imgsearch/internal/embedder"
	"imgsearch/internal/embedder/sqliteai"
)

type sqliteAIEmbedderOptions struct {
	ModelPath          string
	ModelOptions       string
	VisionModelPath    string
	VisionModelOptions string
	ContextOptions     string
}

func newSQLiteAIEmbedder(sqlDB *sql.DB, opts sqliteAIEmbedderOptions) (embedder.Embedder, error) {
	if sqlDB == nil {
		return nil, fmt.Errorf("sqlite-ai embedder requires a db connection")
	}

	modelPath := strings.TrimSpace(opts.ModelPath)
	if modelPath == "" {
		return nil, fmt.Errorf("sqlite-ai model path is required (set -sqlite-ai-model-path)")
	}
	if err := ensureFileExists(modelPath); err != nil {
		return nil, fmt.Errorf("sqlite-ai model path: %w", err)
	}

	visionPath := strings.TrimSpace(opts.VisionModelPath)
	if visionPath == "" {
		return nil, fmt.Errorf("sqlite-ai vision model path is required for image embedding (set -sqlite-ai-vision-model-path)")
	}
	if err := ensureFileExists(visionPath); err != nil {
		return nil, fmt.Errorf("sqlite-ai vision model path: %w", err)
	}

	return sqliteai.New(sqliteai.Config{
		DB:                 sqlDB,
		ModelPath:          modelPath,
		ModelOptions:       strings.TrimSpace(opts.ModelOptions),
		VisionModelPath:    visionPath,
		VisionModelOptions: strings.TrimSpace(opts.VisionModelOptions),
		ContextOptions:     strings.TrimSpace(opts.ContextOptions),
	})
}

func ensureFileExists(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	if info.IsDir() {
		return fmt.Errorf("%q is a directory", path)
	}
	return nil
}
