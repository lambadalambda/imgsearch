package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"imgsearch/internal/annotation"
	"imgsearch/internal/embedder"
	"imgsearch/internal/embedder/openaicompat"
	"imgsearch/internal/settings"
)

func TestResolverOutputTokensLiveChangesWithoutRebuild(t *testing.T) {
	for _, backend := range []string{settings.BackendNative, settings.BackendOpenAI} {
		t.Run(backend, func(t *testing.T) {
			f := newResolverFixture(t)
			ctx := context.Background()
			s := settings.DefaultAnnotation()
			if backend == settings.BackendOpenAI {
				s = remoteSettings("model").Normalized()
			}
			f.save(t, s)
			call := func(want annotation.OutputTokens) {
				t.Helper()
				// Resolver settings override caller-supplied values, without changing metadata.
				opts := embedder.ImageAnnotationOptions{OriginalName: "named", KnownTags: []string{"tag"}, ImageMaxSideMultiplier: 2, OutputTokens: embedder.AnnotationTokenBudget{Primary: 99, Retry: 99}}
				if _, err := f.resolver.AnnotateImageWithOptions(ctx, "image", opts); err != nil {
					t.Fatal(err)
				}
				if _, err := f.resolver.AnnotateVideoFrame(ctx, "frame", opts); err != nil {
					t.Fatal(err)
				}
				if _, err := f.resolver.AnnotateVideo(ctx, embedder.VideoAnnotationInput{OriginalName: "clip"}); err != nil {
					t.Fatal(err)
				}
				a := f.native
				if backend == settings.BackendOpenAI {
					a = f.remote
				}
				image := a.optionsCalls[len(a.optionsCalls)-1]
				frame := a.frameOptions[len(a.frameOptions)-1]
				video := a.videoInputs[len(a.videoInputs)-1]
				if image.OutputTokens != want.ImageBudget() || frame.OutputTokens != want.VideoFrameBudget() || video.OutputTokens != want.VideoBudget() {
					t.Fatalf("budgets image=%+v frame=%+v video=%+v want=%+v", image.OutputTokens, frame.OutputTokens, video.OutputTokens, want)
				}
				if image.OriginalName != opts.OriginalName || frame.ImageMaxSideMultiplier != 2 || len(image.KnownTags) != 1 || video.OriginalName != "clip" {
					t.Fatal("request metadata lost")
				}
				if opts.OutputTokens.Primary != 99 {
					t.Fatal("caller options mutated")
				}
			}
			call(annotation.DefaultOutputTokens())
			s.OutputTokens = annotation.OutputTokens{Image: 2000, ImageRetry: 700, VideoFrame: 500, VideoFrameRetry: 350, Video: 3000, VideoRetry: 900}
			f.save(t, s)
			f.clock = f.clock.Add(6 * time.Second)
			call(s.OutputTokens)
			// A status-triggered refresh also applies budgets without rebuilding.
			s.OutputTokens.Image = 2200
			f.save(t, s)
			if _, err := f.resolver.Status(ctx); err != nil {
				t.Fatal(err)
			}
			call(s.OutputTokens)
			if len(f.nativeBuilds)+len(f.remoteBuilds) != 1 {
				t.Fatalf("budget change rebuilt backend: native=%v remote=%v", f.nativeBuilds, f.remoteBuilds)
			}
		})
	}
}

func TestResolverOutputTokensKeepsAcquiredSnapshot(t *testing.T) {
	f := newResolverFixture(t)
	ctx := context.Background()
	old, err := f.resolver.acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer f.resolver.release()
	saved := settings.DefaultAnnotation()
	saved.OutputTokens.Image = 3000
	saved.OutputTokens.ImageRetry = 900
	f.save(t, saved)
	status, err := f.resolver.Status(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if status.SettingsVersion != 1 {
		t.Fatalf("status: %+v", status)
	}
	current, err := f.resolver.resolve(ctx, false)
	if err != nil {
		t.Fatal(err)
	}
	if old == current || old.annotator != current.annotator {
		t.Fatal("must copy snapshot, not rebuild backend")
	}
	if old.outputTokens != annotation.DefaultOutputTokens() || current.outputTokens != saved.OutputTokens {
		t.Fatalf("old=%+v new=%+v", old.outputTokens, current.outputTokens)
	}
	if f.resolver.draining {
		t.Fatal("budget-only save should not drain active requests")
	}
}

// A real remote request remains on its original pair through a parse retry
// while a settings save publishes a new snapshot for the next request.
func TestResolverOutputTokensInFlightRetryUsesOriginalBudget(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	entered := make(chan struct{}, 1)
	release := make(chan struct{})
	budgets := make(chan int, 4)
	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			MaxTokens int `json:"max_tokens"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			http.Error(w, "bad JSON", 400)
			return
		}
		budgets <- body.MaxTokens
		n := requests.Add(1)
		if n == 1 {
			entered <- struct{}{}
			select {
			case <-release:
			case <-ctx.Done():
				return
			}
		}
		content := `{"title":"Cat","full_description":"A cat","tags":["cat"]}`
		if n%2 == 1 {
			content = "not JSON"
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": content}}}})
	}))
	defer func() { cancel(); srv.Close() }()
	conn := openResolverDB(t)
	s := remoteSettings("model").Normalized()
	s.OpenAI.BaseURL = srv.URL
	if _, err := settings.SaveAnnotation(ctx, conn, s); err != nil {
		t.Fatal(err)
	}
	builds := 0
	resolver := newAnnotatorResolver(annotatorResolverOptions{DB: conn, Defaults: settings.DefaultAnnotation(),
		BuildRemote: func(_ context.Context, s settings.AnnotationSettings) (embedder.ImageAnnotator, settings.ActiveAnnotation, error) {
			builds++
			cfg := openAIConfigFromSettings(s, 512, nil)
			cfg.PrepareImage = func(context.Context, string, int) ([]byte, string, error) { return []byte("image"), "image/jpeg", nil }
			a, err := openaicompat.New(cfg)
			return a, remoteAnnotationStatus(s), err
		},
	})
	defer resolver.Close()
	done := make(chan error, 1)
	go func() { _, err := resolver.AnnotateImage(ctx, "image.jpg"); done <- err }()
	select {
	case <-entered:
	case <-ctx.Done():
		t.Fatal("request never reached server")
	}
	s.OutputTokens.Image = 2400
	s.OutputTokens.ImageRetry = 850
	if _, err := settings.SaveAnnotation(ctx, conn, s); err != nil {
		t.Fatal(err)
	}
	if _, err := resolver.Status(ctx); err != nil {
		t.Fatal(err)
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if _, err := resolver.AnnotateImage(ctx, "image.jpg"); err != nil {
		t.Fatal(err)
	}
	if builds != 1 {
		t.Fatalf("budget-only save rebuilt remote: %d", builds)
	}
	for _, want := range []int{1024, 512, 2400, 850} {
		select {
		case got := <-budgets:
			if got != want {
				t.Fatalf("wire max_tokens=%d want=%d", got, want)
			}
		case <-ctx.Done():
			t.Fatal("missing attempt")
		}
	}
}
