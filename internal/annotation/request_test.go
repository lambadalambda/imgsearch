package annotation

import (
	"testing"

	"imgsearch/internal/embedder"
)

func TestRequestConstructorsPairPromptsWithBudgets(t *testing.T) {
	img := ImageRequest("kid a album cover.jpg", nil)
	if img.SystemPrompt != ImageSystemPrompt || img.JSONSchema != FullJSONSchema || img.MaxTokens != ImageMaxTokens || img.RetryMaxTokens != ImageRetryMaxTokens {
		t.Fatalf("unexpected image request: %+v", img)
	}
	if img.UserPrompt != ImageUserPrompt("kid a album cover.jpg", nil) {
		t.Fatal("image user prompt mismatch")
	}
	frame := VideoFrameRequest("x.mp4", nil)
	if frame.JSONSchema != CompactJSONSchema || frame.MaxTokens != VideoFrameMaxTokens || frame.RetryUserPrompt != VideoFrameRetryUserPrompt {
		t.Fatalf("unexpected frame request: %+v", frame)
	}
	vid, err := VideoRequest(embedder.VideoAnnotationInput{Frames: []embedder.VideoFrameAnnotation{{Description: "d"}}})
	if err != nil {
		t.Fatal(err)
	}
	if vid.JSONSchema != FullJSONSchema || vid.MaxTokens != VideoMaxTokens || vid.SystemPrompt != VideoSystemPrompt {
		t.Fatalf("unexpected video request: %+v", vid)
	}
	if _, err := VideoRequest(embedder.VideoAnnotationInput{}); err == nil {
		t.Fatal("expected error without frames")
	}
}
