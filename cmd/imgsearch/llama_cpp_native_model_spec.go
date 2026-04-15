package main

import (
	"fmt"

	"imgsearch/internal/db"
)

func llamaNativeEmbeddingModelSpec(dimensions int) (db.EmbeddingModelSpec, error) {
	if dimensions <= 0 {
		return db.EmbeddingModelSpec{}, fmt.Errorf("llama-cpp-native expects positive dimensions, got %d", dimensions)
	}

	return db.EmbeddingModelSpec{
		Name:       "llama.cpp-embedding",
		Version:    "native",
		Dimensions: dimensions,
		Metric:     "cosine",
		Normalized: true,
	}, nil
}
