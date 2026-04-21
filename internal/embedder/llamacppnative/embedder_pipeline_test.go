//go:build cgo

package llamacppnative

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

func TestRunEmbedPipelinePreservesOrderAndCleansUp(t *testing.T) {
	paths := []string{"a", "b", "c"}

	var mu sync.Mutex
	cleanupCount := make(map[string]int, len(paths))

	preprocess := func(_ context.Context, path string) (preparedEmbedPath, error) {
		return preparedEmbedPath{
			path: path,
			cleanup: func() {
				mu.Lock()
				cleanupCount[path]++
				mu.Unlock()
			},
		}, nil
	}

	embed := func(_ context.Context, prepared preparedEmbedPath) ([]float32, error) {
		switch prepared.path {
		case "a":
			return []float32{1}, nil
		case "b":
			return []float32{2}, nil
		case "c":
			return []float32{3}, nil
		default:
			t.Fatalf("unexpected prepared path %q", prepared.path)
			return nil, nil
		}
	}

	vecs, err := runEmbedPipeline(context.Background(), paths, preprocess, embed)
	if err != nil {
		t.Fatalf("run pipeline: %v", err)
	}
	if len(vecs) != len(paths) {
		t.Fatalf("vector count: got=%d want=%d", len(vecs), len(paths))
	}
	for i, want := range []float32{1, 2, 3} {
		if len(vecs[i]) != 1 || vecs[i][0] != want {
			t.Fatalf("vector %d: got=%v want=[%v]", i, vecs[i], want)
		}
	}

	mu.Lock()
	defer mu.Unlock()
	for _, path := range paths {
		if cleanupCount[path] != 1 {
			t.Fatalf("cleanup count for %q: got=%d want=1", path, cleanupCount[path])
		}
	}
}

func TestRunEmbedPipelineOverlapsPreprocessAndEmbed(t *testing.T) {
	paths := []string{"p1", "p2"}

	embedFirstStarted := make(chan struct{})
	preprocessSecondStarted := make(chan struct{})
	allowEmbedFirstFinish := make(chan struct{})

	preprocess := func(_ context.Context, path string) (preparedEmbedPath, error) {
		if path == "p2" {
			close(preprocessSecondStarted)
		}
		return preparedEmbedPath{path: path, cleanup: func() {}}, nil
	}

	embed := func(_ context.Context, prepared preparedEmbedPath) ([]float32, error) {
		if prepared.path == "p1" {
			close(embedFirstStarted)
			<-allowEmbedFirstFinish
		}
		return []float32{1}, nil
	}

	errCh := make(chan error, 1)
	go func() {
		_, err := runEmbedPipeline(context.Background(), paths, preprocess, embed)
		errCh <- err
	}()

	select {
	case <-embedFirstStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("first embed did not start")
	}

	select {
	case <-preprocessSecondStarted:
		// Expected: preprocess for next image starts while current embed is in progress.
	case <-time.After(2 * time.Second):
		t.Fatal("second preprocess did not start while first embed was running")
	}

	close(allowEmbedFirstFinish)

	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("run pipeline: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("pipeline did not complete")
	}
}

func TestRunEmbedPipelineCancelsPendingPreprocessedItemOnEmbedError(t *testing.T) {
	paths := []string{"p1", "p2"}
	errBoom := errors.New("boom")

	var mu sync.Mutex
	cleanupCount := map[string]int{"p1": 0, "p2": 0}
	preprocessSecondReady := make(chan struct{})

	preprocess := func(_ context.Context, path string) (preparedEmbedPath, error) {
		if path == "p2" {
			close(preprocessSecondReady)
		}
		return preparedEmbedPath{
			path: path,
			cleanup: func() {
				mu.Lock()
				cleanupCount[path]++
				mu.Unlock()
			},
		}, nil
	}

	embed := func(_ context.Context, prepared preparedEmbedPath) ([]float32, error) {
		if prepared.path != "p1" {
			t.Fatalf("unexpected embed path %q", prepared.path)
		}
		<-preprocessSecondReady
		return nil, errBoom
	}

	_, err := runEmbedPipeline(context.Background(), paths, preprocess, embed)
	if !errors.Is(err, errBoom) {
		t.Fatalf("run pipeline error: got=%v want=%v", err, errBoom)
	}

	mu.Lock()
	defer mu.Unlock()
	if cleanupCount["p1"] != 1 {
		t.Fatalf("cleanup count for p1: got=%d want=1", cleanupCount["p1"])
	}
	if cleanupCount["p2"] != 1 {
		t.Fatalf("cleanup count for p2: got=%d want=1", cleanupCount["p2"])
	}
}

func TestRunEmbedPipelineReturnsPreprocessError(t *testing.T) {
	paths := []string{"p1", "p2"}
	errPreprocess := errors.New("preprocess failed")

	embedStarted := make(chan struct{})
	preprocess := func(_ context.Context, path string) (preparedEmbedPath, error) {
		if path == "p2" {
			return preparedEmbedPath{}, errPreprocess
		}
		return preparedEmbedPath{path: path, cleanup: func() {}}, nil
	}
	embed := func(_ context.Context, prepared preparedEmbedPath) ([]float32, error) {
		close(embedStarted)
		if prepared.path != "p1" {
			t.Fatalf("unexpected embed path %q", prepared.path)
		}
		return []float32{1}, nil
	}

	_, err := runEmbedPipeline(context.Background(), paths, preprocess, embed)
	if !errors.Is(err, errPreprocess) {
		t.Fatalf("run pipeline error: got=%v want=%v", err, errPreprocess)
	}

	select {
	case <-embedStarted:
		// One in-flight embed may complete before the preprocess error is observed.
	default:
		t.Fatal("expected first embed to start before preprocess error")
	}
}

func TestRunEmbedPipelineReturnsContextCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	paths := []string{"p1", "p2"}

	preprocess := func(_ context.Context, path string) (preparedEmbedPath, error) {
		return preparedEmbedPath{path: path, cleanup: func() {}}, nil
	}
	embed := func(_ context.Context, prepared preparedEmbedPath) ([]float32, error) {
		if prepared.path == "p1" {
			cancel()
		}
		return []float32{1}, nil
	}

	_, err := runEmbedPipeline(ctx, paths, preprocess, embed)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("run pipeline error: got=%v want=%v", err, context.Canceled)
	}
}

func TestRunEmbedChunkFallsBackToSerialOnPreprocessError(t *testing.T) {
	paths := []string{"p1", "p2"}
	errPrefetch := errors.New("prefetch failed")

	var mu sync.Mutex
	cleanupCount := map[string]int{"p1": 0, "p2": 0}
	preprocessCalls := 0
	embedCalls := 0
	failedP2Once := false

	preprocess := func(_ context.Context, path string) (preparedEmbedPath, error) {
		preprocessCalls++
		if path == "p2" && !failedP2Once {
			failedP2Once = true
			return preparedEmbedPath{}, errPrefetch
		}
		return preparedEmbedPath{
			path: path,
			cleanup: func() {
				mu.Lock()
				cleanupCount[path]++
				mu.Unlock()
			},
		}, nil
	}
	embed := func(_ context.Context, prepared preparedEmbedPath) ([]float32, error) {
		embedCalls++
		switch prepared.path {
		case "p1":
			return []float32{1}, nil
		case "p2":
			return []float32{2}, nil
		default:
			t.Fatalf("unexpected path %q", prepared.path)
			return nil, nil
		}
	}

	vecs, err := runEmbedChunk(context.Background(), paths, preprocess, embed)
	if err != nil {
		t.Fatalf("run embed chunk: %v", err)
	}
	if len(vecs) != 2 || len(vecs[0]) != 1 || len(vecs[1]) != 1 || vecs[0][0] != 1 || vecs[1][0] != 2 {
		t.Fatalf("unexpected vectors: %v", vecs)
	}
	if preprocessCalls != 4 {
		t.Fatalf("preprocess calls: got=%d want=4", preprocessCalls)
	}
	if embedCalls != 3 {
		t.Fatalf("embed calls: got=%d want=3", embedCalls)
	}

	mu.Lock()
	defer mu.Unlock()
	if cleanupCount["p1"] != 2 {
		t.Fatalf("cleanup count for p1: got=%d want=2", cleanupCount["p1"])
	}
	if cleanupCount["p2"] != 1 {
		t.Fatalf("cleanup count for p2: got=%d want=1", cleanupCount["p2"])
	}
}

func TestRunEmbedChunkReturnsEmbedErrorWithoutFallback(t *testing.T) {
	paths := []string{"p1", "p2"}
	errEmbed := errors.New("embed failed")

	preprocessCalls := 0
	embedCalls := 0

	preprocess := func(_ context.Context, path string) (preparedEmbedPath, error) {
		preprocessCalls++
		return preparedEmbedPath{path: path, cleanup: func() {}}, nil
	}
	embed := func(_ context.Context, prepared preparedEmbedPath) ([]float32, error) {
		embedCalls++
		if prepared.path == "p1" {
			return nil, errEmbed
		}
		return []float32{2}, nil
	}

	_, err := runEmbedChunk(context.Background(), paths, preprocess, embed)
	if !errors.Is(err, errEmbed) {
		t.Fatalf("run embed chunk error: got=%v want=%v", err, errEmbed)
	}
	if preprocessCalls < 1 || preprocessCalls > 2 {
		t.Fatalf("preprocess calls: got=%d want in [1,2]", preprocessCalls)
	}
	if embedCalls != 1 {
		t.Fatalf("embed calls: got=%d want=1", embedCalls)
	}
}

func TestRunEmbedSerialReturnsContextCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := runEmbedSerial(
		ctx,
		[]string{"p1"},
		func(_ context.Context, path string) (preparedEmbedPath, error) {
			return preparedEmbedPath{path: path, cleanup: func() {}}, nil
		},
		func(_ context.Context, prepared preparedEmbedPath) ([]float32, error) {
			return []float32{1}, nil
		},
	)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("run embed serial error: got=%v want=%v", err, context.Canceled)
	}
}
