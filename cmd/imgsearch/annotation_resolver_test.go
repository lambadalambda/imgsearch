package main

import (
	"context"
	"database/sql"
	"errors"
	"strings"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
	"imgsearch/internal/embedder"
	"imgsearch/internal/settings"
)

func openResolverDB(t *testing.T) *sql.DB {
	t.Helper()
	conn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	conn.SetMaxOpenConns(1)
	t.Cleanup(func() { _ = conn.Close() })
	if err := db.RunMigrations(context.Background(), conn); err != nil {
		t.Fatal(err)
	}
	return conn
}

type resolverFixture struct {
	db           *sql.DB
	resolver     *annotatorResolver
	nativeBuilds []string
	remoteBuilds []string
	native       *fakeSwitchAnnotator
	remote       *fakeSwitchAnnotator
	clock        time.Time
	failNative   bool
}

func newResolverFixture(t *testing.T) *resolverFixture {
	t.Helper()
	f := &resolverFixture{db: openResolverDB(t), clock: time.Unix(1_700_000_000, 0)}
	f.resolver = newAnnotatorResolver(annotatorResolverOptions{
		DB:       f.db,
		Defaults: settings.DefaultAnnotation(),
		BuildNative: func(_ context.Context, variant string) (embedder.ImageAnnotator, settings.ActiveAnnotation, error) {
			f.nativeBuilds = append(f.nativeBuilds, variant)
			if f.failNative {
				return nil, settings.ActiveAnnotation{}, errors.New("native load failed")
			}
			f.native = &fakeSwitchAnnotator{id: "native-" + variant}
			return f.native, settings.ActiveAnnotation{Backend: settings.BackendNative, Model: variant}, nil
		},
		BuildRemote: func(_ context.Context, s settings.AnnotationSettings) (embedder.ImageAnnotator, settings.ActiveAnnotation, error) {
			f.remoteBuilds = append(f.remoteBuilds, s.OpenAI.Model)
			f.remote = &fakeSwitchAnnotator{id: "remote-" + s.OpenAI.Model}
			return f.remote, settings.ActiveAnnotation{Backend: settings.BackendOpenAI, Model: s.OpenAI.Model, Detail: s.OpenAI.BaseURL}, nil
		},
		CheckInterval: 5 * time.Second,
		Now:           func() time.Time { return f.clock },
	})
	return f
}

func (f *resolverFixture) save(t *testing.T, s settings.AnnotationSettings) {
	t.Helper()
	if _, err := settings.SaveAnnotation(context.Background(), f.db, s); err != nil {
		t.Fatal(err)
	}
}

func remoteSettings(model string) settings.AnnotationSettings {
	return settings.AnnotationSettings{Backend: settings.BackendOpenAI, OpenAI: settings.OpenAISettings{BaseURL: "http://srv/v1", Model: model}}
}

func TestResolverUsesDefaultsThenSwitchesToRemoteAfterSave(t *testing.T) {
	f := newResolverFixture(t)
	ctx := context.Background()

	if _, err := f.resolver.AnnotateImage(ctx, "a.jpg"); err != nil {
		t.Fatal(err)
	}
	if len(f.nativeBuilds) != 1 || f.nativeBuilds[0] != settings.NativeVariantE4B || len(f.native.imageCalls) != 1 {
		t.Fatalf("expected default native e4b build and call, got builds=%v", f.nativeBuilds)
	}

	f.save(t, remoteSettings("llava"))
	// Within the check interval the resolver keeps the current backend.
	if _, err := f.resolver.AnnotateImage(ctx, "b.jpg"); err != nil {
		t.Fatal(err)
	}
	if len(f.remoteBuilds) != 0 || len(f.native.imageCalls) != 2 {
		t.Fatalf("expected throttled check to keep native, remoteBuilds=%v nativeCalls=%d", f.remoteBuilds, len(f.native.imageCalls))
	}

	f.clock = f.clock.Add(6 * time.Second)
	if _, err := f.resolver.AnnotateImage(ctx, "c.jpg"); err != nil {
		t.Fatal(err)
	}
	if len(f.remoteBuilds) != 1 || f.remoteBuilds[0] != "llava" || len(f.remote.imageCalls) != 1 {
		t.Fatalf("expected remote build and call after interval, builds=%v", f.remoteBuilds)
	}
	if f.native.closeCount != 0 {
		t.Fatal("resolver must not close native annotators; the switchboard owns them")
	}

	status, err := f.resolver.Status(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if status.Backend != settings.BackendOpenAI || status.Model != "llava" || status.SettingsVersion != 1 {
		t.Fatalf("unexpected status: %+v", status)
	}
}

func TestResolverStatusForcesVersionCheckAndSkipsRebuildWhenUnchanged(t *testing.T) {
	f := newResolverFixture(t)
	ctx := context.Background()
	f.save(t, settings.AnnotationSettings{Backend: settings.BackendNative, NativeVariant: settings.NativeVariant26B})

	status, err := f.resolver.Status(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if status.Model != settings.NativeVariant26B || len(f.nativeBuilds) != 1 {
		t.Fatalf("expected 26b build via status, got %+v builds=%v", status, f.nativeBuilds)
	}

	// Saving identical settings bumps the version but must not rebuild.
	f.save(t, settings.AnnotationSettings{Backend: settings.BackendNative, NativeVariant: settings.NativeVariant26B})
	if _, err := f.resolver.Status(ctx); err != nil {
		t.Fatal(err)
	}
	if len(f.nativeBuilds) != 1 {
		t.Fatalf("expected no rebuild for unchanged backend, got builds=%v", f.nativeBuilds)
	}

	f.save(t, settings.AnnotationSettings{Backend: settings.BackendNative, NativeVariant: settings.NativeVariantE4B})
	if _, err := f.resolver.Status(ctx); err != nil {
		t.Fatal(err)
	}
	if len(f.nativeBuilds) != 2 || f.nativeBuilds[1] != settings.NativeVariantE4B {
		t.Fatalf("expected rebuild for changed variant, got %v", f.nativeBuilds)
	}
}

func TestResolverClosesReplacedRemoteAndKeepsWorkingAfterBuildFailure(t *testing.T) {
	f := newResolverFixture(t)
	ctx := context.Background()
	f.save(t, remoteSettings("one"))
	if _, err := f.resolver.AnnotateImage(ctx, "a.jpg"); err != nil {
		t.Fatal(err)
	}
	firstRemote := f.remote

	f.save(t, remoteSettings("two"))
	f.clock = f.clock.Add(6 * time.Second)
	if _, err := f.resolver.AnnotateImage(ctx, "b.jpg"); err != nil {
		t.Fatal(err)
	}
	if firstRemote.closeCount != 1 {
		t.Fatalf("expected replaced remote annotator to be closed, got %d", firstRemote.closeCount)
	}

	f.failNative = true
	f.save(t, settings.AnnotationSettings{Backend: settings.BackendNative, NativeVariant: settings.NativeVariant26B})
	f.clock = f.clock.Add(6 * time.Second)
	_, err := f.resolver.AnnotateImage(ctx, "c.jpg")
	if err == nil || !strings.Contains(err.Error(), "native load failed") {
		t.Fatalf("expected build failure to surface, got %v", err)
	}
	status, statusErr := f.resolver.Status(ctx)
	if statusErr == nil {
		t.Fatalf("expected status to report the build failure, got %+v", status)
	}
	// Throttled calls fail consistently instead of falling back to the
	// replaced backend, and the old remote client was released.
	if _, err := f.resolver.AnnotateImage(ctx, "c2.jpg"); err == nil {
		t.Fatal("expected throttled call after a failed swap to error too")
	}
	if len(f.remote.imageCalls) != 1 || f.remote.closeCount != 1 {
		t.Fatalf("expected replaced remote closed and unused: calls=%d closes=%d", len(f.remote.imageCalls), f.remote.closeCount)
	}
	// Once the backend builds again, the resolver recovers without a restart.
	f.failNative = false
	f.clock = f.clock.Add(6 * time.Second)
	if _, err := f.resolver.AnnotateImage(ctx, "d.jpg"); err != nil {
		t.Fatalf("expected recovery after builder succeeds: %v", err)
	}
	if got, _ := f.resolver.Status(ctx); got.Source != settings.ActiveSourceWorker {
		t.Fatalf("expected worker-sourced status, got %+v", got)
	}
	if err := f.resolver.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestResolverDelegatesVideoMethodsAndRejectsUnsupported(t *testing.T) {
	f := newResolverFixture(t)
	ctx := context.Background()
	if _, err := f.resolver.AnnotateVideoFrame(ctx, "f.jpg", embedder.ImageAnnotationOptions{OriginalName: "x"}); err != nil {
		t.Fatal(err)
	}
	if _, err := f.resolver.AnnotateVideo(ctx, embedder.VideoAnnotationInput{}); err != nil {
		t.Fatal(err)
	}
	if len(f.native.frameCalls) != 1 || f.native.videoCallCount != 1 {
		t.Fatalf("expected delegation: frames=%v videos=%d", f.native.frameCalls, f.native.videoCallCount)
	}

	plain := newAnnotatorResolver(annotatorResolverOptions{
		DB:       f.db,
		Defaults: settings.DefaultAnnotation(),
		BuildNative: func(context.Context, string) (embedder.ImageAnnotator, settings.ActiveAnnotation, error) {
			return imageOnlyAnnotator{}, settings.ActiveAnnotation{Backend: settings.BackendNative}, nil
		},
	})
	if _, err := plain.AnnotateVideo(ctx, embedder.VideoAnnotationInput{}); err == nil {
		t.Fatal("expected error for annotator without video support")
	}
	if _, err := plain.AnnotateImageWithOptions(ctx, "a.jpg", embedder.ImageAnnotationOptions{}); err != nil {
		t.Fatalf("plain annotator should still serve images: %v", err)
	}
}

type imageOnlyAnnotator struct{}

func (imageOnlyAnnotator) AnnotateImage(context.Context, string) (embedder.ImageAnnotation, error) {
	return embedder.ImageAnnotation{Description: "plain"}, nil
}
