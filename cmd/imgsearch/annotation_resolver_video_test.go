package main

import (
	"context"
	"testing"
	"time"

	"imgsearch/internal/embedder"
	"imgsearch/internal/settings"
)

func TestResolverPinsVideoModeAndBackendThroughJob(t *testing.T) {
	for _, startRemote := range []bool{false, true} {
		t.Run(map[bool]string{false: "native_to_direct", true: "direct_to_native"}[startRemote], func(t *testing.T) {
			f := newResolverFixture(t)
			ctx := context.Background()
			direct := remoteSettings("vision")
			direct.VideoMode = "direct_frames"
			native := settings.DefaultAnnotation()
			// The saved mode must not turn native into direct input.
			native.VideoMode = "direct_frames"
			if startRemote {
				f.save(t, direct)
			} else {
				f.save(t, native)
			}
			job, release, err := f.resolver.AcquireVideoAnnotator(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer release()
			old := f.native
			wantMode := "frame_descriptions"
			if startRemote {
				old = f.remote
				wantMode = "direct_frames"
			}
			if job.VideoMode() != wantMode {
				t.Fatalf("mode=%s", job.VideoMode())
			}
			if startRemote {
				f.save(t, native)
			} else {
				f.save(t, direct)
			}
			f.clock = f.clock.Add(6 * time.Second)
			if _, err = f.resolver.Status(ctx); err != nil {
				t.Fatal(err)
			}
			if n, err := f.resolver.AnnotationConcurrency(ctx); err != nil || n != 0 {
				t.Fatalf("admission=%d err=%v", n, err)
			}
			if _, err = job.AnnotateVideoFrame(ctx, "frame.jpg", embedder.ImageAnnotationOptions{}); err != nil {
				t.Fatal(err)
			}
			if _, err = job.AnnotateVideo(ctx, embedder.VideoAnnotationInput{}); err != nil {
				t.Fatal(err)
			}
			if job.VideoMode() != wantMode || len(old.frameCalls) != 1 || old.videoCallCount != 1 || old.closeCount != 0 {
				t.Fatalf("unpinned job mode=%s old=%+v", job.VideoMode(), old)
			}
			release()
			next, done, err := f.resolver.AcquireVideoAnnotator(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer done()
			if next.VideoMode() == wantMode {
				t.Fatalf("next job kept old mode %s", wantMode)
			}
			if startRemote && old.closeCount != 1 {
				t.Fatalf("old remote not closed after job: %d", old.closeCount)
			}
		})
	}
}

func TestResolverRemoteVideoModeChangeRebuilds(t *testing.T) {
	f := newResolverFixture(t)
	s := remoteSettings("vision")
	f.save(t, s)
	if _, err := f.resolver.Status(context.Background()); err != nil {
		t.Fatal(err)
	}
	s.VideoMode = "direct_frames"
	f.save(t, s)
	if _, err := f.resolver.Status(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(f.remoteBuilds) != 2 {
		t.Fatalf("remote builds=%v", f.remoteBuilds)
	}
	if got := openAIConfigFromSettings(s, 384, nil).VideoMode; got != "direct_frames" {
		t.Fatalf("remote config mode=%s", got)
	}
}

func TestAcquiredVideoJobKeepsBudgetsAcrossSettingsRefresh(t *testing.T) {
	for _, mode := range []string{"frame_descriptions", "direct_frames"} {
		t.Run(mode, func(t *testing.T) {
			f := newResolverFixture(t)
			ctx := context.Background()
			s := remoteSettings("vision").Normalized()
			s.VideoMode = mode
			f.save(t, s)
			job, release, err := f.resolver.AcquireVideoAnnotator(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer release()
			old := s.OutputTokens
			s.OutputTokens.VideoFrame = 800
			s.OutputTokens.VideoFrameRetry = 400
			s.OutputTokens.Video = 2400
			s.OutputTokens.VideoRetry = 900
			f.save(t, s)
			if _, err := f.resolver.Status(ctx); err != nil {
				t.Fatal(err)
			}
			check := func(job embedder.VideoJobAnnotator, frame, video embedder.AnnotationTokenBudget) {
				t.Helper()
				if _, err := job.AnnotateVideoFrame(ctx, "frame", embedder.ImageAnnotationOptions{}); err != nil {
					t.Fatal(err)
				}
				if _, err := job.AnnotateVideo(ctx, embedder.VideoAnnotationInput{}); err != nil {
					t.Fatal(err)
				}
				a := f.remote
				if got := a.frameOptions[len(a.frameOptions)-1].OutputTokens; got != frame {
					t.Fatalf("frame budget %+v want%+v", got, frame)
				}
				if got := a.videoInputs[len(a.videoInputs)-1].OutputTokens; got != video {
					t.Fatalf("video budget %+v want%+v", got, video)
				}
			}
			check(job, old.VideoFrameBudget(), old.VideoBudget())
			release()
			next, done, err := f.resolver.AcquireVideoAnnotator(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer done()
			check(next, s.OutputTokens.VideoFrameBudget(), s.OutputTokens.VideoBudget())
			if len(f.remoteBuilds) != 1 {
				t.Fatal("budget-only change rebuilt backend")
			}
		})
	}
}
