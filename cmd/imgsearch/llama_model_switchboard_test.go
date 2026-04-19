package main

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"imgsearch/internal/embedder"
)

type fakeSwitchEmbedder struct {
	id         string
	closeCount int
	textCalls  []string
	imageCalls []string
	batchCalls [][]string
}

func (f *fakeSwitchEmbedder) Close() error {
	f.closeCount++
	return nil
}

func (f *fakeSwitchEmbedder) EmbedText(_ context.Context, text string) ([]float32, error) {
	f.textCalls = append(f.textCalls, text)
	return []float32{1, 2, 3}, nil
}

func (f *fakeSwitchEmbedder) EmbedImage(_ context.Context, path string) ([]float32, error) {
	f.imageCalls = append(f.imageCalls, path)
	return []float32{4, 5, 6}, nil
}

func (f *fakeSwitchEmbedder) EmbedImages(_ context.Context, paths []string) ([][]float32, error) {
	chunk := make([]string, len(paths))
	copy(chunk, paths)
	f.batchCalls = append(f.batchCalls, chunk)
	out := make([][]float32, len(paths))
	for i := range paths {
		out[i] = []float32{float32(i + 1)}
	}
	return out, nil
}

type fakeSwitchAnnotator struct {
	id             string
	closeCount     int
	imageCalls     []string
	optionsCalls   []embedder.ImageAnnotationOptions
	videoCallCount int
}

func (f *fakeSwitchAnnotator) Close() error {
	f.closeCount++
	return nil
}

func (f *fakeSwitchAnnotator) AnnotateImage(_ context.Context, path string) (embedder.ImageAnnotation, error) {
	f.imageCalls = append(f.imageCalls, path)
	return embedder.ImageAnnotation{Description: "desc", Tags: []string{"tag"}}, nil
}

func (f *fakeSwitchAnnotator) AnnotateImageWithOptions(_ context.Context, path string, opts embedder.ImageAnnotationOptions) (embedder.ImageAnnotation, error) {
	f.imageCalls = append(f.imageCalls, path)
	f.optionsCalls = append(f.optionsCalls, opts)
	return embedder.ImageAnnotation{Description: "desc", Tags: []string{"tag"}}, nil
}

func (f *fakeSwitchAnnotator) AnnotateVideo(_ context.Context, _ embedder.VideoAnnotationInput) (embedder.VideoAnnotation, error) {
	f.videoCallCount++
	return embedder.VideoAnnotation{Description: "video", Tags: []string{"video-tag"}}, nil
}

func TestLlamaModelSwitchboardSwapsModelsOnDemand(t *testing.T) {
	ctx := context.Background()
	initialEmbedder := &fakeSwitchEmbedder{id: "embedder-1"}
	embedderLoads := 1
	annotatorLoads := 0
	loadedEmbedders := []*fakeSwitchEmbedder{initialEmbedder}
	loadedAnnotators := []*fakeSwitchAnnotator{}

	switchboard := newLlamaModelSwitchboard(
		initialEmbedder,
		func(context.Context) (embedder.Embedder, error) {
			embedderLoads++
			instance := &fakeSwitchEmbedder{id: fmt.Sprintf("embedder-%d", embedderLoads)}
			loadedEmbedders = append(loadedEmbedders, instance)
			return instance, nil
		},
		func(context.Context) (embedder.ImageAnnotator, error) {
			annotatorLoads++
			instance := &fakeSwitchAnnotator{id: fmt.Sprintf("annotator-%d", annotatorLoads)}
			loadedAnnotators = append(loadedAnnotators, instance)
			return instance, nil
		},
		true,
	)
	t.Cleanup(func() { _ = switchboard.Close() })

	if _, err := switchboard.Embedder().EmbedText(ctx, "first query"); err != nil {
		t.Fatalf("EmbedText with initial embedder: %v", err)
	}
	if initialEmbedder.closeCount != 0 {
		t.Fatalf("initial embedder should stay loaded, close_count=%d", initialEmbedder.closeCount)
	}

	if _, err := switchboard.Annotator().AnnotateImage(ctx, "/tmp/image.jpg"); err != nil {
		t.Fatalf("AnnotateImage after swap: %v", err)
	}
	if initialEmbedder.closeCount != 1 {
		t.Fatalf("expected embedder to unload before annotator load, close_count=%d", initialEmbedder.closeCount)
	}
	if annotatorLoads != 1 {
		t.Fatalf("expected one annotator load, got %d", annotatorLoads)
	}

	if _, err := switchboard.Embedder().EmbedImage(ctx, "/tmp/second.jpg"); err != nil {
		t.Fatalf("EmbedImage after swapping back: %v", err)
	}
	if len(loadedAnnotators) != 1 || loadedAnnotators[0].closeCount != 1 {
		t.Fatalf("expected annotator to unload when switching back, annotators=%d close_count=%d", len(loadedAnnotators), loadedAnnotators[0].closeCount)
	}
	if embedderLoads != 2 {
		t.Fatalf("expected embedder to reload once, got load_count=%d", embedderLoads)
	}
	if len(loadedEmbedders) < 2 || len(loadedEmbedders[1].imageCalls) != 1 {
		t.Fatalf("expected reloaded embedder to serve EmbedImage call")
	}
}

func TestLlamaModelSwitchboardSupportsBatchAndVideoAnnotator(t *testing.T) {
	ctx := context.Background()
	initialEmbedder := &fakeSwitchEmbedder{id: "embedder-1"}
	annotator := &fakeSwitchAnnotator{id: "annotator-1"}

	switchboard := newLlamaModelSwitchboard(
		initialEmbedder,
		func(context.Context) (embedder.Embedder, error) {
			return initialEmbedder, nil
		},
		func(context.Context) (embedder.ImageAnnotator, error) {
			return annotator, nil
		},
		true,
	)
	t.Cleanup(func() { _ = switchboard.Close() })

	batcher, ok := switchboard.Embedder().(embedder.BatchImageEmbedder)
	if !ok {
		t.Fatalf("switching embedder should implement BatchImageEmbedder")
	}
	vecs, err := batcher.EmbedImages(ctx, []string{"a.jpg", "b.jpg"})
	if err != nil {
		t.Fatalf("EmbedImages: %v", err)
	}
	if len(vecs) != 2 {
		t.Fatalf("expected 2 vectors from batch embed, got %d", len(vecs))
	}

	withOpts, ok := switchboard.Annotator().(embedder.ImageAnnotatorWithOptions)
	if !ok {
		t.Fatalf("switching annotator should implement ImageAnnotatorWithOptions")
	}
	if _, err := withOpts.AnnotateImageWithOptions(ctx, "img.jpg", embedder.ImageAnnotationOptions{OriginalName: "named.jpg"}); err != nil {
		t.Fatalf("AnnotateImageWithOptions: %v", err)
	}
	if len(annotator.optionsCalls) != 1 || annotator.optionsCalls[0].OriginalName != "named.jpg" {
		t.Fatalf("expected options passthrough to underlying annotator")
	}

	videoAnnotator, ok := switchboard.Annotator().(embedder.VideoAnnotator)
	if !ok {
		t.Fatalf("switching annotator should implement VideoAnnotator when enabled")
	}
	if _, err := videoAnnotator.AnnotateVideo(ctx, embedder.VideoAnnotationInput{RepresentativeFramePath: "frame.jpg", Frames: []embedder.VideoFrameAnnotation{{Description: "frame"}}}); err != nil {
		t.Fatalf("AnnotateVideo: %v", err)
	}
	if annotator.videoCallCount != 1 {
		t.Fatalf("expected video annotation passthrough call count 1, got %d", annotator.videoCallCount)
	}
}

func TestLlamaModelSwitchboardClosePreventsFurtherUse(t *testing.T) {
	initialEmbedder := &fakeSwitchEmbedder{id: "embedder-1"}
	switchboard := newLlamaModelSwitchboard(
		initialEmbedder,
		func(context.Context) (embedder.Embedder, error) {
			return initialEmbedder, nil
		},
		func(context.Context) (embedder.ImageAnnotator, error) {
			return &fakeSwitchAnnotator{id: "annotator-1"}, nil
		},
		true,
	)

	if err := switchboard.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	_, err := switchboard.Embedder().EmbedText(context.Background(), "q")
	if err == nil || !strings.Contains(strings.ToLower(err.Error()), "closed") {
		t.Fatalf("expected closed error from embedder after switchboard close, got %v", err)
	}
}
