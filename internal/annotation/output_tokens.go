package annotation

import (
	"fmt"

	"imgsearch/internal/embedder"
)

// OutputTokens holds output-generation limits, not image preprocessing or
// context-window sizes. Zero fields in older persisted settings use defaults.
type OutputTokens struct {
	Image           int `json:"image"`
	ImageRetry      int `json:"image_retry"`
	VideoFrame      int `json:"video_frame"`
	VideoFrameRetry int `json:"video_frame_retry"`
	Video           int `json:"video"`
	VideoRetry      int `json:"video_retry"`
}

// Leave room for at least a compact JSON response, while bounding accidental
// runaway generation. These limits do not guarantee a model's context can fit
// the requested output, or that generation will succeed.
const (
	MinOutputTokens = 64
	MaxOutputTokens = 32768
)

func DefaultOutputTokens() OutputTokens {
	return OutputTokens{
		Image:           ImageMaxTokens,
		ImageRetry:      ImageRetryMaxTokens,
		VideoFrame:      VideoFrameMaxTokens,
		VideoFrameRetry: VideoFrameRetryMaxTokens,
		Video:           VideoMaxTokens,
		VideoRetry:      VideoRetryMaxTokens,
	}
}

func (b OutputTokens) Normalized() OutputTokens {
	defaults := DefaultOutputTokens()
	if b.Image == 0 {
		b.Image = defaults.Image
	}
	if b.ImageRetry == 0 {
		b.ImageRetry = defaults.ImageRetry
	}
	if b.VideoFrame == 0 {
		b.VideoFrame = defaults.VideoFrame
	}
	if b.VideoFrameRetry == 0 {
		b.VideoFrameRetry = defaults.VideoFrameRetry
	}
	if b.Video == 0 {
		b.Video = defaults.Video
	}
	if b.VideoRetry == 0 {
		b.VideoRetry = defaults.VideoRetry
	}
	return b
}

// Validate is strict: callers must fill omitted fields before validation.
func (b OutputTokens) Validate() error {
	for _, field := range []struct {
		name  string
		value int
	}{
		{"image", b.Image},
		{"image_retry", b.ImageRetry},
		{"video_frame", b.VideoFrame},
		{"video_frame_retry", b.VideoFrameRetry},
		{"video", b.Video},
		{"video_retry", b.VideoRetry},
	} {
		if field.value < MinOutputTokens || field.value > MaxOutputTokens {
			return fmt.Errorf("output_tokens.%s must be between %d and %d", field.name, MinOutputTokens, MaxOutputTokens)
		}
	}
	return nil
}

func (b OutputTokens) ImageBudget() embedder.AnnotationTokenBudget {
	return embedder.AnnotationTokenBudget{Primary: b.Image, Retry: b.ImageRetry}
}
func (b OutputTokens) VideoFrameBudget() embedder.AnnotationTokenBudget {
	return embedder.AnnotationTokenBudget{Primary: b.VideoFrame, Retry: b.VideoFrameRetry}
}
func (b OutputTokens) VideoBudget() embedder.AnnotationTokenBudget {
	return embedder.AnnotationTokenBudget{Primary: b.Video, Retry: b.VideoRetry}
}
