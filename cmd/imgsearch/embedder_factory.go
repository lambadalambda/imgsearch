package main

import (
	"fmt"

	"imgsearch/internal/db"
	"imgsearch/internal/embedder"
	"imgsearch/internal/embedder/deterministic"
	"imgsearch/internal/embedder/jinamlx"
)

func embedderDimensionsForType(kind string) (int, error) {
	switch kind {
	case "deterministic", "jina-mlx", "jina-torch":
		return 2048, nil
	case "qwen3-vl-embedding-8b":
		return 4096, nil
	default:
		return 0, fmt.Errorf("unsupported embedder %q", kind)
	}
}

func newEmbedder(kind string, jinaURL string, dimensions int, imageMode string) (embedder.Embedder, error) {
	switch kind {
	case "deterministic":
		return &deterministic.Embedder{Dimensions: dimensions}, nil
	case "jina-mlx", "jina-torch":
		if jinaURL == "" {
			return nil, fmt.Errorf("%s sidecar URL is required (set -jina-mlx-url flag)", kind)
		}
		if dimensions != 2048 {
			return nil, fmt.Errorf("%s expects 2048 dimensions, got %d", kind, dimensions)
		}
		if !isSupportedImageMode(imageMode) {
			return nil, fmt.Errorf("unsupported embed image mode %q (expected path, bytes, or auto)", imageMode)
		}
		return jinamlx.NewHTTPClientWithImageMode(jinaURL, imageMode), nil
	case "qwen3-vl-embedding-8b":
		if jinaURL == "" {
			return nil, fmt.Errorf("%s sidecar URL is required (set -jina-mlx-url flag)", kind)
		}
		if dimensions != 4096 {
			return nil, fmt.Errorf("%s expects 4096 dimensions, got %d", kind, dimensions)
		}
		if !isSupportedImageMode(imageMode) {
			return nil, fmt.Errorf("unsupported embed image mode %q (expected path, bytes, or auto)", imageMode)
		}
		return jinamlx.NewHTTPClientWithImageMode(jinaURL, imageMode), nil
	default:
		return nil, fmt.Errorf("unsupported embedder %q", kind)
	}
}

func isSupportedImageMode(mode string) bool {
	switch mode {
	case "path", "bytes", "auto":
		return true
	default:
		return false
	}
}

func embeddingModelSpec(kind string, dimensions int) (db.EmbeddingModelSpec, error) {
	switch kind {
	case "deterministic":
		return db.EmbeddingModelSpec{
			Name:       "deterministic-sha256",
			Version:    "v1",
			Dimensions: dimensions,
			Metric:     "cosine",
			Normalized: true,
		}, nil
	case "jina-mlx":
		if dimensions != 2048 {
			return db.EmbeddingModelSpec{}, fmt.Errorf("jina-mlx expects 2048 dimensions, got %d", dimensions)
		}
		return db.EmbeddingModelSpec{
			Name:       "jina-embeddings-v4",
			Version:    "mlx-8bit",
			Dimensions: dimensions,
			Metric:     "cosine",
			Normalized: true,
		}, nil
	case "jina-torch":
		if dimensions != 2048 {
			return db.EmbeddingModelSpec{}, fmt.Errorf("jina-torch expects 2048 dimensions, got %d", dimensions)
		}
		return db.EmbeddingModelSpec{
			Name:       "jina-embeddings-v4",
			Version:    "torch",
			Dimensions: dimensions,
			Metric:     "cosine",
			Normalized: true,
		}, nil
	case "qwen3-vl-embedding-8b":
		if dimensions != 4096 {
			return db.EmbeddingModelSpec{}, fmt.Errorf("qwen3-vl-embedding-8b expects 4096 dimensions, got %d", dimensions)
		}
		return db.EmbeddingModelSpec{
			Name:       "Qwen3-VL-Embedding-8B",
			Version:    "transformers",
			Dimensions: dimensions,
			Metric:     "cosine",
			Normalized: true,
		}, nil
	default:
		return db.EmbeddingModelSpec{}, fmt.Errorf("unsupported embedder %q", kind)
	}
}
