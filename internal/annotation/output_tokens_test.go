package annotation

import (
	"testing"

	"imgsearch/internal/embedder"
)

func TestRequestOutputTokenBudgets(t *testing.T) {
	video, err := VideoRequest(embedder.VideoAnnotationInput{Frames: []embedder.VideoFrameAnnotation{{Description: "frame"}}, OutputTokens: embedder.AnnotationTokenBudget{Primary: 2500, Retry: 750}})
	if err != nil {
		t.Fatal(err)
	}
	if video.MaxTokens != 2500 || video.RetryMaxTokens != 750 {
		t.Fatalf("video input budget: %+v", video)
	}
	for _, req := range []Request{ImageRequest("", nil), VideoFrameRequest("", nil), video} {
		original := req
		if got := req.WithOutputTokens(embedder.AnnotationTokenBudget{}); got != req {
			t.Fatalf("zero budget changed request: %+v", got)
		}
		got := req.WithOutputTokens(embedder.AnnotationTokenBudget{Primary: 2048, Retry: 900})
		if got.MaxTokens != 2048 || got.RetryMaxTokens != 900 {
			t.Fatalf("budget not applied: %+v", got)
		}
		got.MaxTokens = req.MaxTokens
		got.RetryMaxTokens = req.RetryMaxTokens
		if got != original || req != original {
			t.Fatal("budget application mutated prompts or original request")
		}
		if got := req.WithOutputTokens(embedder.AnnotationTokenBudget{Retry: 800}); got.MaxTokens != req.MaxTokens || got.RetryMaxTokens != 800 {
			t.Fatalf("partial pair: %+v", got)
		}
	}
}
