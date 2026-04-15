package main

import (
	"fmt"
	"os"
	"strings"

	"imgsearch/internal/embedder"
	"imgsearch/internal/embedder/llamacppnative"
)

type llamaCPPNativeEmbedderOptions struct {
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

const defaultLlamaNativeImageMaxSide = 384

func resolveLlamaCPPNativeImageLimits(imageMaxSide int, imageMaxTokens int) (int, int, error) {
	resolvedImageMaxSide := imageMaxSide
	if resolvedImageMaxSide == 0 {
		resolvedImageMaxSide = defaultLlamaNativeImageMaxSide
	}
	if resolvedImageMaxSide < 0 {
		return 0, 0, fmt.Errorf("llama-cpp-native image max side must be non-negative")
	}
	if imageMaxTokens < 0 {
		return 0, 0, fmt.Errorf("llama-cpp-native image max tokens must be non-negative")
	}

	return resolvedImageMaxSide, imageMaxTokens, nil
}

func newLlamaCPPNativeEmbedder(opts llamaCPPNativeEmbedderOptions) (embedder.Embedder, error) {
	modelPath := strings.TrimSpace(opts.ModelPath)
	if modelPath == "" {
		return nil, fmt.Errorf("llama-cpp-native model path is required (set -llama-native-model-path)")
	}
	if err := ensureRegularFile(modelPath); err != nil {
		return nil, fmt.Errorf("llama-cpp-native model path: %w", err)
	}

	visionPath := strings.TrimSpace(opts.VisionModelPath)
	if visionPath == "" {
		return nil, fmt.Errorf("llama-cpp-native vision model path is required (set -llama-native-mmproj-path)")
	}
	if err := ensureRegularFile(visionPath); err != nil {
		return nil, fmt.Errorf("llama-cpp-native vision model path: %w", err)
	}

	if opts.Dimensions <= 0 {
		return nil, fmt.Errorf("llama-cpp-native dimensions must be positive")
	}

	if opts.ContextSize <= 0 {
		return nil, fmt.Errorf("llama-cpp-native context size must be positive")
	}

	if opts.BatchSize <= 0 {
		return nil, fmt.Errorf("llama-cpp-native batch size must be positive")
	}

	imageMaxSide, imageMaxTokens, err := resolveLlamaCPPNativeImageLimits(opts.ImageMaxSide, opts.ImageMaxTokens)
	if err != nil {
		return nil, err
	}

	return llamacppnative.New(llamacppnative.Config{
		ModelPath:          modelPath,
		VisionModelPath:    visionPath,
		Dimensions:         opts.Dimensions,
		GPULayers:          opts.GPULayers,
		UseGPU:             opts.UseGPU,
		ContextSize:        opts.ContextSize,
		BatchSize:          opts.BatchSize,
		Threads:            opts.Threads,
		ImageMaxSide:       imageMaxSide,
		ImageMaxTokens:     imageMaxTokens,
		QueryInstruction:   strings.TrimSpace(opts.QueryInstruction),
		PassageInstruction: strings.TrimSpace(opts.PassageInstruction),
	})
}

func ensureRegularFile(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	if info.IsDir() {
		return fmt.Errorf("%q is a directory", path)
	}
	return nil
}
