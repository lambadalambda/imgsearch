//go:build !llamacpp_native

package llamacppnative

import (
	"context"
	"fmt"
)

type Config struct {
	ModelPath          string
	VisionModelPath    string
	Dimensions         int
	GPULayers          int
	UseGPU             bool
	ContextSize        int
	BatchSize          int
	Threads            int
	ImageMaxSide       int
	ImageMaxTokens     int
	QueryInstruction   string
	PassageInstruction string
}

type Embedder struct{}

func New(Config) (*Embedder, error) {
	return nil, fmt.Errorf("llama-cpp-native requires build tag 'llamacpp_native' and a built llama.cpp runtime")
}

func (e *Embedder) Close() error { return nil }

func (e *Embedder) EmbedText(context.Context, string) ([]float32, error) {
	return nil, fmt.Errorf("llama-cpp-native embedder is unavailable in this build")
}

func (e *Embedder) EmbedImage(context.Context, string) ([]float32, error) {
	return nil, fmt.Errorf("llama-cpp-native embedder is unavailable in this build")
}
