package embedder

import "context"

type TextEmbedder interface {
	EmbedText(ctx context.Context, text string) ([]float32, error)
}

type ImageEmbedder interface {
	EmbedImage(ctx context.Context, path string) ([]float32, error)
}

type Embedder interface {
	TextEmbedder
	ImageEmbedder
}
