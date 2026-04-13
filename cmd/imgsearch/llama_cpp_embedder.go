package main

import (
	"fmt"
	"strings"

	"imgsearch/internal/embedder"
	"imgsearch/internal/embedder/llamacpp"
)

type llamaCPPEmbedderOptions struct {
	URL                string
	Dimensions         int
	Model              string
	QueryInstruction   string
	PassageInstruction string
}

func newLlamaCPPEmbedder(opts llamaCPPEmbedderOptions) (embedder.Embedder, error) {
	url := strings.TrimSpace(opts.URL)
	if url == "" {
		return nil, fmt.Errorf("llama-cpp embedder URL is required (set -llama-cpp-url)")
	}
	if opts.Dimensions <= 0 {
		return nil, fmt.Errorf("llama-cpp dimensions must be positive")
	}

	return llamacpp.New(llamacpp.Config{
		BaseURL:            url,
		Dimensions:         opts.Dimensions,
		Model:              strings.TrimSpace(opts.Model),
		QueryInstruction:   strings.TrimSpace(opts.QueryInstruction),
		PassageInstruction: strings.TrimSpace(opts.PassageInstruction),
	})
}
