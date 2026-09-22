package main

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"imgsearch/internal/embedder"
	"imgsearch/internal/settings"
)

type blockingResolverAnnotator struct {
	entered chan struct{}
	release chan struct{}
	closed  atomic.Int32
}

func (a *blockingResolverAnnotator) AnnotateImage(ctx context.Context, _ string) (embedder.ImageAnnotation, error) {
	a.entered <- struct{}{}
	select {
	case <-a.release:
		return embedder.ImageAnnotation{}, nil
	case <-ctx.Done():
		return embedder.ImageAnnotation{}, ctx.Err()
	}
}
func (a *blockingResolverAnnotator) Close() error { a.closed.Add(1); return nil }

func TestResolverSwapWaitsForActiveCalls(t *testing.T) {
	conn := openResolverDB(t)
	old := &blockingResolverAnnotator{entered: make(chan struct{}, 2), release: make(chan struct{})}
	var builds atomic.Int32
	r := newAnnotatorResolver(annotatorResolverOptions{DB: conn, Defaults: settings.DefaultAnnotation(), CheckInterval: time.Millisecond,
		BuildRemote: func(_ context.Context, s settings.AnnotationSettings) (embedder.ImageAnnotator, settings.ActiveAnnotation, error) {
			builds.Add(1)
			return old, settings.ActiveAnnotation{Backend: settings.BackendOpenAI, Model: s.OpenAI.Model}, nil
		},
		BuildNative: func(context.Context, string) (embedder.ImageAnnotator, settings.ActiveAnnotation, error) {
			builds.Add(1)
			return &fakeSwitchAnnotator{}, settings.ActiveAnnotation{Backend: settings.BackendNative}, nil
		},
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if _, err := settings.SaveAnnotation(ctx, conn, remoteSettings("old")); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 2)
	for range 2 {
		go func() { _, err := r.AnnotateImage(ctx, "x"); done <- err }()
	}
	for range 2 {
		select {
		case <-old.entered:
		case <-time.After(time.Second):
			t.Fatal("two remote calls did not overlap")
		}
	}
	if _, err := settings.SaveAnnotation(ctx, conn, settings.DefaultAnnotation()); err != nil {
		t.Fatal(err)
	}
	status, err := r.Status(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if old.closed.Load() != 0 || builds.Load() != 1 {
		t.Fatalf("swap closed/built while active: closed=%d builds=%d", old.closed.Load(), builds.Load())
	}
	if status.Backend != settings.BackendOpenAI {
		t.Fatal("active status changed before drain")
	}
	close(old.release)
	for range 2 {
		if err := <-done; err != nil {
			t.Fatal(err)
		}
	}
	status, err = r.Status(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if status.Backend != settings.BackendNative || old.closed.Load() != 1 {
		t.Fatalf("swap did not finish: %+v closed=%d", status, old.closed.Load())
	}
}

func TestResolverPersistedConcurrencyAndCloseDrain(t *testing.T) {
	conn := openResolverDB(t)
	makeBackend := func() *blockingResolverAnnotator {
		return &blockingResolverAnnotator{entered: make(chan struct{}, 8), release: make(chan struct{})}
	}
	var backend *blockingResolverAnnotator
	r := newAnnotatorResolver(annotatorResolverOptions{DB: conn, Defaults: settings.DefaultAnnotation(), CheckInterval: time.Millisecond,
		BuildRemote: func(_ context.Context, s settings.AnnotationSettings) (embedder.ImageAnnotator, settings.ActiveAnnotation, error) {
			backend = makeBackend()
			return backend, settings.ActiveAnnotation{Backend: settings.BackendOpenAI, Model: s.OpenAI.Model}, nil
		},
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	save := func(limit int) {
		t.Helper()
		s := remoteSettings("test")
		s.OpenAI.Concurrency = limit
		if _, err := settings.SaveAnnotation(ctx, conn, s); err != nil {
			t.Fatal(err)
		}
		if _, err := r.Status(ctx); err != nil {
			t.Fatal(err)
		}
	}
	limitIs := func(want int) {
		t.Helper()
		got, err := r.AnnotationConcurrency(ctx)
		if err != nil || got != want {
			t.Fatalf("concurrency=%d err=%v want=%d", got, err, want)
		}
	}
	save(2)
	limitIs(2)
	old := backend
	done := make(chan error, 4)
	call := func() { go func() { _, err := r.AnnotateImage(ctx, "x"); done <- err }() }
	entered := func(a *blockingResolverAnnotator) {
		t.Helper()
		select {
		case <-a.entered:
		case <-time.After(time.Second):
			t.Fatal("call did not start")
		}
	}
	call()
	call()
	entered(old)
	entered(old)
	save(1)
	limitIs(0)
	if old.closed.Load() != 0 {
		t.Fatal("closed before calls drained")
	}
	close(old.release)
	for range 2 {
		if err := <-done; err != nil {
			t.Fatal(err)
		}
	}
	limitIs(1)
	if old.closed.Load() != 1 {
		t.Fatal("old backend was not closed exactly once")
	}
	current := backend
	call()
	entered(current)
	call()
	select {
	case <-current.entered:
		t.Fatal("configured one was not serial")
	case <-time.After(50 * time.Millisecond):
	}
	// Close prevents further admission, but must not close an active resource.
	closed := make(chan struct{})
	go func() { _ = r.Close(); close(closed) }()
	select {
	case <-closed:
		t.Fatal("Close returned with active call")
	case <-time.After(50 * time.Millisecond):
	}
	if current.closed.Load() != 0 {
		t.Fatal("active backend closed")
	}
	cancel()
	for range 2 {
		if err := <-done; err == nil {
			t.Fatal("expected canceled/closed call")
		}
	}
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("Close did not drain")
	}
	if current.closed.Load() != 1 {
		t.Fatal("backend close count wrong")
	}
	if _, err := r.AnnotateImage(context.Background(), "x"); err == nil {
		t.Fatal("closed resolver accepted work")
	}
}

func TestResolverNativeAdmissionIsSerial(t *testing.T) {
	conn := openResolverDB(t)
	a := &blockingResolverAnnotator{entered: make(chan struct{}, 2), release: make(chan struct{})}
	r := newAnnotatorResolver(annotatorResolverOptions{DB: conn, Defaults: settings.DefaultAnnotation(), BuildNative: func(context.Context, string) (embedder.ImageAnnotator, settings.ActiveAnnotation, error) {
		return a, settings.ActiveAnnotation{Backend: settings.BackendNative}, nil
	}})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 2)
	for range 2 {
		go func() { _, err := r.AnnotateImage(ctx, "x"); done <- err }()
	}
	select {
	case <-a.entered:
	case <-time.After(time.Second):
		t.Fatal("native call did not start")
	}
	select {
	case <-a.entered:
		t.Fatal("native calls overlapped")
	case <-time.After(50 * time.Millisecond):
	}
	cancel()
	for range 2 {
		if err := <-done; err == nil {
			t.Fatal("expected cancellation")
		}
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if a.closed.Load() != 0 {
		t.Fatal("resolver closed switchboard-owned native annotator")
	}
}
