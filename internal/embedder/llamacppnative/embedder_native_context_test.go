//go:build cgo

package llamacppnative

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestEmbedderMethodsRespectCanceledContextBeforeNativeWork(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	e := &Embedder{}

	if _, err := e.EmbedText(ctx, "hello"); !errors.Is(err, context.Canceled) {
		t.Fatalf("EmbedText error: got=%v want=%v", err, context.Canceled)
	}
	if _, err := e.EmbedImage(ctx, "ignored.jpg"); !errors.Is(err, context.Canceled) {
		t.Fatalf("EmbedImage error: got=%v want=%v", err, context.Canceled)
	}
}

func TestEmbedderMethodsReturnClosedErrorWithActiveContext(t *testing.T) {
	e := &Embedder{}

	if _, err := e.EmbedText(context.Background(), "hello"); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("EmbedText error: got=%v want closed error", err)
	}
}
