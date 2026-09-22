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
	Title       string
	Summary     string
	Description string
	Tags        []string
}

// AnnotationTokenBudget is copied into each request; zero values retain the
// annotation kind's defaults. It never changes the loaded model configuration.
type AnnotationTokenBudget struct {
	Primary int
	Retry   int
}

type ImageAnnotationOptions struct {
	OutputTokens           AnnotationTokenBudget
	OriginalName           string
	ImageMaxSideMultiplier int
	// KnownTags is the library's existing tag vocabulary, most used first;
	// prompts ask the model to prefer these over inventing near-synonyms.
	KnownTags []string
}

type ImageAnnotator interface {
	AnnotateImage(ctx context.Context, path string) (ImageAnnotation, error)
}

type ImageAnnotatorWithOptions interface {
	ImageAnnotator
	AnnotateImageWithOptions(ctx context.Context, path string, opts ImageAnnotationOptions) (ImageAnnotation, error)
}

type VideoFrameAnnotator interface {
	AnnotateVideoFrame(ctx context.Context, path string, opts ImageAnnotationOptions) (ImageAnnotation, error)
}

type VideoFrameAnnotation struct {
	FrameIndex  int
	TimestampMS int64
	Description string
	Tags        []string
}

type VideoAnnotationInput struct {
	OutputTokens            AnnotationTokenBudget
	OriginalName            string
	DurationMS              int64
	RepresentativeFramePath string
	ImageMaxSideMultiplier  int
	TranscriptText          string
	Frames                  []VideoFrameAnnotation
	// KnownTags mirrors ImageAnnotationOptions.KnownTags.
	KnownTags []string
}

type VideoAnnotation struct {
	Title       string
	Summary     string
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
