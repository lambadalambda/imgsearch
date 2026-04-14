package main

import (
	"fmt"
	"strings"

	"imgsearch/internal/embedder"
	"imgsearch/internal/embedder/llamacppnative"
)

type llamaCPPNativeAnnotatorOptions struct {
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

func newLlamaCPPNativeAnnotator(opts llamaCPPNativeAnnotatorOptions) (embedder.ImageAnnotator, error) {
	modelPath := strings.TrimSpace(opts.ModelPath)
	if modelPath == "" {
		return nil, fmt.Errorf("llama-cpp-native annotator model path is required (set -llama-native-annotator-model-path)")
	}
	if err := ensureRegularFile(modelPath); err != nil {
		return nil, fmt.Errorf("llama-cpp-native annotator model path: %w", err)
	}

	visionPath := strings.TrimSpace(opts.VisionModelPath)
	if visionPath == "" {
		return nil, fmt.Errorf("llama-cpp-native annotator vision model path is required (set -llama-native-annotator-mmproj-path)")
	}
	if err := ensureRegularFile(visionPath); err != nil {
		return nil, fmt.Errorf("llama-cpp-native annotator vision model path: %w", err)
	}

	if opts.ContextSize <= 0 {
		return nil, fmt.Errorf("llama-cpp-native annotator context size must be positive")
	}
	if opts.BatchSize <= 0 {
		return nil, fmt.Errorf("llama-cpp-native annotator batch size must be positive")
	}

	imageMaxSide, imageMaxTokens, err := resolveLlamaCPPNativeImageLimits(opts.ImageMaxSide, opts.ImageMaxTokens)
	if err != nil {
		return nil, err
	}

	return llamacppnative.NewAnnotator(llamacppnative.AnnotatorConfig{
		ModelPath:       modelPath,
		VisionModelPath: visionPath,
		GPULayers:       opts.GPULayers,
		UseGPU:          opts.UseGPU,
		ContextSize:     opts.ContextSize,
		BatchSize:       opts.BatchSize,
		Threads:         opts.Threads,
		ImageMaxSide:    imageMaxSide,
		ImageMaxTokens:  imageMaxTokens,
	})
}
