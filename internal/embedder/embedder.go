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

type ImageAnnotationOptions struct {
	OriginalName string
}

type ImageAnnotator interface {
	AnnotateImage(ctx context.Context, path string) (ImageAnnotation, error)
}

type ImageAnnotatorWithOptions interface {
	ImageAnnotator
	AnnotateImageWithOptions(ctx context.Context, path string, opts ImageAnnotationOptions) (ImageAnnotation, error)
}

type VideoFrameAnnotation struct {
	FrameIndex  int
	TimestampMS int64
	Description string
	Tags        []string
}

type VideoAnnotationInput struct {
	OriginalName            string
	DurationMS              int64
	RepresentativeFramePath string
	TranscriptText          string
	Frames                  []VideoFrameAnnotation
}

type VideoAnnotation struct {
	Description string
	Tags        []string
	IsNSFW      bool
}

type VideoAnnotator interface {
	AnnotateVideo(ctx context.Context, input VideoAnnotationInput) (VideoAnnotation, error)
}

type Embedder interface {
	TextEmbedder
	ImageEmbedder
}
