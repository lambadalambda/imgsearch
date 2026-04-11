package main

import (
	"fmt"

	"imgsearch/internal/db"
	"imgsearch/internal/embedder"
	"imgsearch/internal/embedder/deterministic"
	"imgsearch/internal/embedder/jinamlx"
)

func newEmbedder(kind string, jinaURL string, dimensions int) (embedder.Embedder, error) {
	switch kind {
	case "deterministic":
		return &deterministic.Embedder{Dimensions: dimensions}, nil
	case "jina-mlx":
		if jinaURL == "" {
			return nil, fmt.Errorf("jina-mlx-url is required")
		}
		if dimensions != 2048 {
			return nil, fmt.Errorf("jina-mlx expects 2048 dimensions, got %d", dimensions)
		}
		return jinamlx.NewHTTPClient(jinaURL), nil
	default:
		return nil, fmt.Errorf("unsupported embedder %q", kind)
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
	default:
		return db.EmbeddingModelSpec{}, fmt.Errorf("unsupported embedder %q", kind)
	}
}
