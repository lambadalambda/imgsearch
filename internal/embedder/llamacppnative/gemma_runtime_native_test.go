//go:build cgo

package llamacppnative

import (
	"context"
	"errors"
	"strings"
	"testing"
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
}
