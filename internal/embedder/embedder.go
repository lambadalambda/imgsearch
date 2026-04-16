package embedder

import "context"

type TextEmbedder interface {
	EmbedText(ctx context.Context, text string) ([]float32, error)
}

type ImageEmbedder interface {
	EmbedImage(ctx context.Context, path string) ([]float32, error)
}

type BatchImageEmbedder interface {
	EmbedImages(ctx context.Context, paths []string) ([][]float32, error)
}

type ImageAnnotation struct {
	Description string
	Tags        []string
}

type ImageAnnotator interface {
	AnnotateImage(ctx context.Context, path string) (ImageAnnotation, error)
}

type Embedder interface {
	TextEmbedder
	ImageEmbedder
}
