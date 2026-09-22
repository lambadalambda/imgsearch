package openaicompat

import (
	"context"
	"testing"

	"imgsearch/internal/embedder"
)

func TestAnnotationOutputTokenBudgetsPrimaryAndRetry(t *testing.T) {
	for _, kind := range []string{"image", "video_frame", "video"} {
		for _, custom := range []bool{false, true} {
			name := kind + "/defaults"
			if custom {
				name = kind + "/custom"
			}
			t.Run(name, func(t *testing.T) {
				f := newFakeServer(t, func(n int, _ map[string]any) (int, string) {
					if n%2 == 0 {
						return 200, chatReply("not JSON")
					}
					return 200, chatReply(goodImageJSON)
				})
				a := newAnnotator(t, f, nil)
				pair := embedder.AnnotationTokenBudget{}
				want := map[string]embedder.AnnotationTokenBudget{"image": {Primary: 1024, Retry: 512}, "video_frame": {Primary: 320, Retry: 256}, "video": {Primary: 1200, Retry: 640}}[kind]
				if custom {
					pair = embedder.AnnotationTokenBudget{Primary: 2300, Retry: 800}
					want = pair
				}
				path := writeTempImage(t)
				var err error
				switch kind {
				case "image":
					_, err = a.AnnotateImageWithOptions(context.Background(), path, embedder.ImageAnnotationOptions{OutputTokens: pair})
				case "video_frame":
					_, err = a.AnnotateVideoFrame(context.Background(), path, embedder.ImageAnnotationOptions{OutputTokens: pair})
				case "video":
					_, err = a.AnnotateVideo(context.Background(), embedder.VideoAnnotationInput{RepresentativeFramePath: path, Frames: []embedder.VideoFrameAnnotation{{Description: "frame"}}, OutputTokens: pair})
				}
				if err != nil {
					t.Fatal(err)
				}
				requests := f.chatRequests()
				if len(requests) != 2 {
					t.Fatalf("expected primary + retry, got %d", len(requests))
				}
				for i, budget := range []int{want.Primary, want.Retry} {
					if got := requests[i].Body["max_tokens"]; got != float64(budget) {
						t.Errorf("attempt %d: got %v want %d", i, got, budget)
					}
				}
				// Overrides are request-local, not a mutation of the shared client.
				if _, err := a.AnnotateImage(context.Background(), path); err != nil {
					t.Fatal(err)
				}
				requests = f.chatRequests()
				if requests[2].Body["max_tokens"] != float64(1024) || requests[3].Body["max_tokens"] != float64(512) {
					t.Fatalf("request budget leaked: %v", requests[2:])
				}
			})
		}
	}
}
