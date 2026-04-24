package vectorindex

import (
	"errors"
	"testing"
)

func TestCosineValidatesDimensions(t *testing.T) {
	_, err := Cosine([]float32{1, 0}, []float32{1, 0, 0})
	if !errors.Is(err, ErrVectorDimensionMismatch) {
		t.Fatalf("expected ErrVectorDimensionMismatch, got %v", err)
	}
}

func TestCosineComputesSimilarity(t *testing.T) {
	sim, err := Cosine([]float32{1, 0}, []float32{1, 0})
	if err != nil {
		t.Fatalf("cosine: %v", err)
	}
	if sim != 1 {
		t.Fatalf("similarity: got=%f want=1", sim)
	}
}

func TestCosineTreatsZeroNormAsDistant(t *testing.T) {
	sim, err := Cosine([]float32{0, 0}, []float32{1, 0})
	if err != nil {
		t.Fatalf("cosine: %v", err)
	}
	if sim != 0 {
		t.Fatalf("similarity: got=%f want=0", sim)
	}
}
