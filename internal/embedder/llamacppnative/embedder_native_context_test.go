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
	if _, err := e.EmbedImage(context.Background(), "ignored.jpg"); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("EmbedImage error: got=%v want closed error", err)
	}
	if _, err := e.executePreparedImageLocked(context.Background(), preparedEmbedPath{path: "ignored.jpg", cleanup: func() {}}); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("executePreparedImageLocked error: got=%v want closed error", err)
	}
}

func TestPrepareImageForEmbeddingRejectsEmptyPath(t *testing.T) {
	e := &Embedder{imageMaxSide: defaultImageMaxSide}

	_, err := e.prepareImageForEmbedding(context.Background(), "   ")
	if err == nil || !strings.Contains(err.Error(), "image path is empty") {
		t.Fatalf("prepareImageForEmbedding error: got=%v want empty path error", err)
	}
}

func TestExecutePreparedImageLockedRespectsCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	e := &Embedder{}
	_, err := e.executePreparedImageLocked(ctx, preparedEmbedPath{path: "ignored.jpg", cleanup: func() {}})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("executePreparedImageLocked error: got=%v want=%v", err, context.Canceled)
	}
}

func TestExecutePreparedImageRespectsCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	e := &Embedder{}
	_, err := e.executePreparedImage(ctx, preparedEmbedPath{path: "ignored.jpg", cleanup: func() {}})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("executePreparedImage error: got=%v want=%v", err, context.Canceled)
	}
}
