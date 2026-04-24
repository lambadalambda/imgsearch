package vectorindex

import (
	"fmt"
	"math"
)

func Cosine(a, b []float32) (float64, error) {
	if len(a) == 0 || len(b) == 0 {
		return 0, fmt.Errorf("%w: vectors must not be empty", ErrVectorDimensionMismatch)
	}
	if len(a) != len(b) {
		return 0, fmt.Errorf("%w: got %d and %d", ErrVectorDimensionMismatch, len(a), len(b))
	}

	var dot float64
	var na float64
	var nb float64
	for i := range a {
		av := float64(a[i])
		bv := float64(b[i])
		dot += av * bv
		na += av * av
		nb += bv * bv
	}
	if na == 0 || nb == 0 {
		return 0, nil
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb)), nil
}
