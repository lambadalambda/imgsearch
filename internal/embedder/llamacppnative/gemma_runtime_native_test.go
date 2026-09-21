//go:build cgo

package llamacppnative

import (
	"context"
	"errors"
	"strings"
	"testing"

	coreembedder "imgsearch/internal/embedder"
)

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
	if _, err := runtime.AnnotateVideoFrame(ctx, "ignored.jpg", coreembedder.ImageAnnotationOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("AnnotateVideoFrame error: got=%v want=%v", err, context.Canceled)
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
	if _, err := runtime.AnnotateVideoFrame(context.Background(), "ignored.jpg", coreembedder.ImageAnnotationOptions{}); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("AnnotateVideoFrame error: got=%v want closed error", err)
	}
}

func TestFormatGenerationTimingLogIncludesNativeBreakdown(t *testing.T) {
	line := formatGenerationTimingLog("annotate_image", "primary", "photo.jpg", generationTiming{
		ImagePreprocessMS:         12,
		NativeDecodeMS:            34,
		TokenizeMS:                5,
		PrefillMS:                 67,
		GenerateMS:                890,
		GeneratedTokens:           321,
		PromptTokens:              654,
		SpeculativeDraftedTokens:  17,
		SpeculativeAcceptedTokens: 9,
	})

	for _, want := range []string{
		"native annotation timing kind=annotate_image",
		"attempt=primary",
		"file=\"photo.jpg\"",
		"preprocess=12ms",
		"native_decode=34ms",
		"tokenize=5ms",
		"prefill=67ms",
		"generate=890ms",
		"generated_tokens=321",
		"prompt_tokens=654",
		"speculative_drafted_tokens=17",
		"speculative_accepted_tokens=9",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("expected %q in %q", want, line)
		}
	}
}
