package deterministic

import (
	"context"
	"crypto/sha256"
	"math"
	"os"
)

type Embedder struct {
	Dimensions int
}

func (e *Embedder) EmbedText(_ context.Context, text string) ([]float32, error) {
	return embedBytes([]byte(text), e.dimensions()), nil
}

func (e *Embedder) EmbedImage(_ context.Context, path string) ([]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return embedBytes(b, e.dimensions()), nil
}

func (e *Embedder) dimensions() int {
	if e.Dimensions <= 0 {
		return 2048
	}
	return e.Dimensions
}

func embedBytes(data []byte, dim int) []float32 {
	v := make([]float32, dim)
	state := sha256.Sum256(data)
	for i := 0; i < dim; i++ {
		if i%32 == 0 {
			state = sha256.Sum256(state[:])
		}
		b := state[i%32]
		v[i] = (float32(b)/127.5 - 1.0)
	}
	norm := float32(0)
	for _, x := range v {
		norm += x * x
	}
	if norm == 0 {
		return v
	}
	inv := float32(1.0 / math.Sqrt(float64(norm)))
	for i := range v {
		v[i] *= inv
	}
	return v
}
