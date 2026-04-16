//go:build !cgo

package llamacppnative

import (
	"context"
	"fmt"

	coreembedder "imgsearch/internal/embedder"
)

type Config struct {
	ModelPath          string
	VisionModelPath    string
	Dimensions         int
	GPULayers          int
	UseGPU             bool
	ContextSize        int
	BatchSize          int
	MaxSequences       int
	Threads            int
	ImageMaxSide       int
	ImageMaxTokens     int
	QueryInstruction   string
	PassageInstruction string
}

type Embedder struct{}

type AnnotatorConfig struct {
	ModelPath       string
	VisionModelPath string
	GPULayers       int
	UseGPU          bool
	ContextSize     int
	BatchSize       int
	Threads         int
	ImageMaxSide    int
	ImageMaxTokens  int
}

type Annotator struct{}

type EmbedInspect struct {
	TextChunks     int
	ImageChunks    int
	TextTokens     int
	ImageTokens    int
	TotalTokens    int
	TotalPositions int
	MaxImageTokens int
	MaxImageNX     int
	MaxImageNY     int
	NCtx           int
	NCtxSeq        int
	NSeqMax        int
	NBatch         int
}

func New(Config) (*Embedder, error) {
	return nil, fmt.Errorf("llama-cpp-native requires cgo and a built llama.cpp runtime")
}

func NewAnnotator(AnnotatorConfig) (*Annotator, error) {
	return nil, fmt.Errorf("llama-cpp-native annotator requires cgo and a built llama.cpp runtime")
}

func (e *Embedder) Close() error { return nil }

func (e *Embedder) EmbedText(context.Context, string) ([]float32, error) {
	return nil, fmt.Errorf("llama-cpp-native embedder is unavailable in this build")
}

func (e *Embedder) EmbedImage(context.Context, string) ([]float32, error) {
	return nil, fmt.Errorf("llama-cpp-native embedder is unavailable in this build")
}

func (e *Embedder) EmbedImages(context.Context, []string) ([][]float32, error) {
	return nil, fmt.Errorf("llama-cpp-native embedder is unavailable in this build")
}

func (e *Embedder) InspectImageEmbedding(context.Context, string) (EmbedInspect, error) {
	return EmbedInspect{}, fmt.Errorf("llama-cpp-native embedder is unavailable in this build")
}

func (a *Annotator) Close() error { return nil }

func (a *Annotator) AnnotateImage(context.Context, string) (coreembedder.ImageAnnotation, error) {
	return coreembedder.ImageAnnotation{}, fmt.Errorf("llama-cpp-native annotator is unavailable in this build")
}
