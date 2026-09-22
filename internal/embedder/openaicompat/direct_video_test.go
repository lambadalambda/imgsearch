package openaicompat

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"imgsearch/internal/embedder"
)

func TestDirectVideoPreservesAllEvidenceAcrossRetries(t *testing.T) {
	f := newFakeServer(t, func(n int, _ map[string]any) (int, string) {
		switch n {
		case 0:
			return 400, `{"error":{"message":"response_format unsupported"}}`
		case 1:
			return 503, `{"error":{"message":"busy"}}`
		case 2:
			return 200, chatReply("not json")
		default:
			return 200, chatReply(`{"title":"Stage","summary":"A show","full_description":"A stage performance","tags":["Stage"],"is_nsfw":true}`)
		}
	})
	var prepared []string
	a := newAnnotator(t, f, func(c *Config) {
		c.VideoMode = "direct_frames"
		c.PrepareImage = func(_ context.Context, path string, maxSide int) ([]byte, string, error) {
			if maxSide != 2048 {
				t.Errorf("maxSide=%d", maxSide)
			}
			prepared = append(prepared, path)
			return []byte(path), "image/jpeg", nil
		}
	})
	input := embedder.VideoAnnotationInput{OriginalName: "stage show.mp4", DurationMS: 12000, TranscriptText: "hello audience", KnownTags: []string{"concert"}, ImageMaxSideMultiplier: 2, OutputTokens: embedder.AnnotationTokenBudget{Primary: 2300, Retry: 900}, Frames: []embedder.VideoFrameAnnotation{
		{Path: "last.jpg", FrameIndex: 2, TimestampMS: 9000},
		{Path: "same.jpg", FrameIndex: 0, TimestampMS: 1000},
		{Path: "same.jpg", FrameIndex: 1, TimestampMS: 4000},
	}}
	got, err := a.AnnotateVideo(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if !got.IsNSFW || !reflect.DeepEqual(got.Tags, []string{"stage", "nsfw"}) {
		t.Fatalf("normalization: %+v", got)
	}
	if !reflect.DeepEqual(prepared, []string{"same.jpg", "same.jpg", "last.jpg"}) {
		t.Fatalf("prepared=%v", prepared)
	}
	requests := f.chatRequests()
	if len(requests) != 4 {
		t.Fatalf("requests=%d", len(requests))
	}
	var evidence []any
	for n, r := range requests {
		parts := r.Body["messages"].([]any)[1].(map[string]any)["content"].([]any)
		if len(parts) != 7 {
			t.Fatalf("request %d parts=%v", n, parts)
		}
		text := parts[0].(map[string]any)["text"].(string)
		for _, want := range []string{"stage show", "12000", "hello audience", "concert", "any sampled frame", "full_description"} {
			if !strings.Contains(text, want) {
				t.Errorf("request %d missing %q", n, want)
			}
		}
		for i, path := range prepared {
			stamp := []int{1000, 4000, 9000}[i]
			label := parts[1+i*2].(map[string]any)["text"].(string)
			if !strings.Contains(label, fmt.Sprintf("timestamp_ms=%d", stamp)) || !strings.Contains(label, fmt.Sprintf("frame_index=%d", i)) {
				t.Errorf("label=%q", label)
			}
			url := parts[2+i*2].(map[string]any)["image_url"].(map[string]any)["url"]
			if url != "data:image/jpeg;base64,"+base64.StdEncoding.EncodeToString([]byte(path)) {
				t.Errorf("url=%v", url)
			}
		}
		if n == 0 {
			evidence = parts[1:]
		} else if !reflect.DeepEqual(evidence, parts[1:]) {
			t.Errorf("retry lost evidence")
		}
		wantTokens := float64(2300)
		if n == 3 {
			wantTokens = 900
		}
		if r.Body["max_tokens"] != wantTokens {
			t.Errorf("tokens=%v", r.Body["max_tokens"])
		}
	}
}

func TestDirectVideoFailsWithoutDroppingFrames(t *testing.T) {
	for _, failure := range []string{"missing path", "unreadable", "provider rejection", "no frames"} {
		t.Run(failure, func(t *testing.T) {
			f := newFakeServer(t, func(int, map[string]any) (int, string) {
				return http.StatusBadRequest, `{"error":{"message":"multiple images unsupported"}}`
			})
			a := newAnnotator(t, f, func(c *Config) {
				c.VideoMode = "direct_frames"
				c.PrepareImage = func(_ context.Context, path string, _ int) ([]byte, string, error) {
					if path == "bad" {
						return nil, "", errors.New("unreadable")
					}
					return []byte(path), "image/jpeg", nil
				}
			})
			input := embedder.VideoAnnotationInput{Frames: []embedder.VideoFrameAnnotation{{Path: "good"}, {Path: "bad"}}}
			switch failure {
			case "missing path":
				input.Frames[1].Path = ""
			case "provider rejection":
				input.Frames[1].Path = "good"
			case "no frames":
				input.Frames = nil
			}
			if _, err := a.AnnotateVideo(context.Background(), input); err == nil {
				t.Fatal("expected error")
			}
			want := 0
			if failure == "provider rejection" {
				want = 1
			}
			if len(f.chatRequests()) != want {
				t.Fatalf("requests=%d", len(f.chatRequests()))
			}
		})
	}
}
