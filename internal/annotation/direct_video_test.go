package annotation

import (
	"strings"
	"testing"

	"imgsearch/internal/embedder"
)

func TestDirectVideoRequestUsesFullSchemaAndRetainsRetryContext(t *testing.T) {
	input := embedder.VideoAnnotationInput{OriginalName: "concert stage.mp4", DurationMS: 7000, TranscriptText: "welcome audience", KnownTags: []string{"concert"}, Frames: []embedder.VideoFrameAnnotation{{Path: "frame.jpg", TimestampMS: 1000, Description: "stale cache"}}}
	req, err := DirectVideoRequest(input)
	if err != nil {
		t.Fatal(err)
	}
	if req.JSONSchema != FullJSONSchema || req.MaxTokens != VideoMaxTokens || req.RetryMaxTokens != VideoRetryMaxTokens {
		t.Fatalf("request=%+v", req)
	}
	if !strings.Contains(req.RetryUserPrompt, req.UserPrompt) || !strings.Contains(req.RetryUserPrompt, VideoRetryUserPrompt) {
		t.Fatal("retry lost original context or compact instruction")
	}
	for _, want := range []string{"concert stage", "7000", "welcome audience", "concert", "any sampled frame", "title", "summary", "full_description", "is_nsfw"} {
		if !strings.Contains(req.UserPrompt, want) {
			t.Errorf("missing %q", want)
		}
	}
	if strings.Contains(req.UserPrompt, "stale cache") || strings.Contains(req.SystemPrompt, "summaries from") {
		t.Fatal("direct prompt uses frame descriptions")
	}
	input.OutputTokens = embedder.AnnotationTokenBudget{Primary: 2300, Retry: 800}
	req, err = DirectVideoRequest(input)
	if err != nil {
		t.Fatal(err)
	}
	if req.MaxTokens != 2300 || req.RetryMaxTokens != 800 {
		t.Fatalf("budgets=%+v", req)
	}
	for _, frames := range [][]embedder.VideoFrameAnnotation{nil, {{Path: "ok"}, {Path: " "}}} {
		input.Frames = frames
		if _, err = DirectVideoRequest(input); err == nil {
			t.Fatal("accepted incomplete frame evidence")
		}
	}
}
