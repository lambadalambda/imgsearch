//go:build cgo

package llamacppnative

import (
	"context"
	"errors"
	"strings"
	"testing"

	coreembedder "imgsearch/internal/embedder"
)

func TestDecodeGemmaJSONObjectStripsMarkdownCodeFences(t *testing.T) {
	raw := "```json\n{\n  \"description\": \"example\",\n  \"tags\": [\"sample\", \"nsfw\"],\n  \"is_nsfw\": true\n}\n```"

	var got gemmaAppAnnotation
	if err := decodeGemmaJSONObject(raw, &got, "description+tags"); err != nil {
		t.Fatalf("decode fenced JSON: %v", err)
	}
	if got.Description != "example" {
		t.Fatalf("unexpected description: %q", got.Description)
	}
	if len(got.Tags) != 2 || got.Tags[0] != "sample" || got.Tags[1] != "nsfw" {
		t.Fatalf("unexpected tags: %v", got.Tags)
	}
	if !got.IsNSFW {
		t.Fatal("expected is_nsfw to be true")
	}
}

func TestStripMarkdownCodeFencesLeavesPlainJSONUntouched(t *testing.T) {
	raw := "{\"description\":\"plain\"}"
	if got := stripMarkdownCodeFences(raw); got != raw {
		t.Fatalf("unexpected stripped value: %q", got)
	}
}

func TestNativeGemmaRuntimeMethodsRespectCanceledContextBeforeNativeWork(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	runtime := &nativeGemmaRuntime{}

	if _, err := runtime.AnnotateImage(ctx, "ignored.jpg"); !errors.Is(err, context.Canceled) {
		t.Fatalf("AnnotateImage error: got=%v want=%v", err, context.Canceled)
	}
	if _, _, err := runtime.DescribeImage(ctx, "ignored.jpg"); !errors.Is(err, context.Canceled) {
		t.Fatalf("DescribeImage error: got=%v want=%v", err, context.Canceled)
	}
	if _, _, err := runtime.AutoTagImage(ctx, "ignored.jpg"); !errors.Is(err, context.Canceled) {
		t.Fatalf("AutoTagImage error: got=%v want=%v", err, context.Canceled)
	}
	if _, _, err := runtime.DescribeAndTagImage(ctx, "ignored.jpg"); !errors.Is(err, context.Canceled) {
		t.Fatalf("DescribeAndTagImage error: got=%v want=%v", err, context.Canceled)
	}
	if _, err := runtime.AnnotateVideo(ctx, coreembedder.VideoAnnotationInput{RepresentativeFramePath: "ignored.jpg"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("AnnotateVideo error: got=%v want=%v", err, context.Canceled)
	}
}

func TestNativeGemmaRuntimeMethodsReturnClosedErrorWithActiveContext(t *testing.T) {
	runtime := &nativeGemmaRuntime{}

	if _, err := runtime.AnnotateImage(context.Background(), "ignored.jpg"); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("AnnotateImage error: got=%v want closed error", err)
	}
	if _, _, err := runtime.DescribeImage(context.Background(), "ignored.jpg"); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("DescribeImage error: got=%v want closed error", err)
	}
	if _, _, err := runtime.AutoTagImage(context.Background(), "ignored.jpg"); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("AutoTagImage error: got=%v want closed error", err)
	}
	if _, _, err := runtime.DescribeAndTagImage(context.Background(), "ignored.jpg"); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("DescribeAndTagImage error: got=%v want closed error", err)
	}
	if _, err := runtime.AnnotateVideo(context.Background(), coreembedder.VideoAnnotationInput{RepresentativeFramePath: "ignored.jpg"}); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("AnnotateVideo error: got=%v want closed error", err)
	}
}

func TestBuildImageAnnotationUserPromptIncludesMeaningfulFilename(t *testing.T) {
	prompt := buildImageAnnotationUserPrompt("the cover of the album kid a by radiohead.jpg")
	if !strings.Contains(prompt, "Original filename") {
		t.Fatalf("expected prompt to include original filename guidance")
	}
	if !strings.Contains(strings.ToLower(prompt), "kid a") {
		t.Fatalf("expected prompt to include meaningful filename content")
	}
	if !strings.Contains(strings.ToLower(prompt), "500 words") {
		t.Fatalf("expected prompt to allow up to 500 words")
	}
	if !strings.Contains(strings.ToLower(prompt), "if there is text in the image") {
		t.Fatalf("expected prompt to explicitly request text description and translation")
	}
}

func TestBuildImageAnnotationUserPromptSkipsNoisyFilename(t *testing.T) {
	prompt := buildImageAnnotationUserPrompt("34254745943.jpg")
	if strings.Contains(prompt, "Original filename") {
		t.Fatalf("expected noisy numeric filename to be omitted from prompt")
	}
}

func TestBuildVideoAnnotationUserPromptIncludesFrameEvidence(t *testing.T) {
	prompt, err := buildVideoAnnotationUserPrompt(coreembedder.VideoAnnotationInput{
		OriginalName:            "party_clip.mp4",
		DurationMS:              12000,
		RepresentativeFramePath: "images/frame-01.jpg",
		TranscriptText:          "hello from the stage",
		Frames: []coreembedder.VideoFrameAnnotation{
			{FrameIndex: 0, TimestampMS: 500, Description: "a singer on stage", Tags: []string{"concert", "singer"}},
			{FrameIndex: 1, TimestampMS: 1500, Description: "crowd waving", Tags: []string{"crowd", "concert"}},
		},
	})
	if err != nil {
		t.Fatalf("buildVideoAnnotationUserPrompt error: %v", err)
	}
	if !strings.Contains(prompt, "Sampled frame annotations") {
		t.Fatalf("expected prompt to include sampled frame annotations section")
	}
	if !strings.Contains(prompt, "hello from the stage") {
		t.Fatalf("expected prompt to include transcript context")
	}
	if !strings.Contains(strings.ToLower(prompt), "meme") || !strings.Contains(strings.ToLower(prompt), "music video") || !strings.Contains(strings.ToLower(prompt), "clip from a show") {
		t.Fatalf("expected prompt to include filename-based media type inference guidance")
	}
	if !strings.Contains(strings.ToLower(prompt), "500 words") {
		t.Fatalf("expected video prompt to allow up to 500 words")
	}
}
